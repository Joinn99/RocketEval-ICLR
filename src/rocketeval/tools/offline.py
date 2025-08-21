import os
import yaml
import json
import logging
import pandas as pd
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from vllm import LLM, SamplingParams

logger = logging.getLogger("rich")

def load_offline_config(offline_config: str) -> dict:
    with open(offline_config, "r") as f:
        return yaml.safe_load(f)

def get_token_ids(
    llm: LLM,
    choices: List[str] = ["Yes", "No"],
    **kwargs
) -> List[int]:
    """
    Get the token id of the choices.
    """
    tokenizer = llm.get_tokenizer()
    choice_tokens = [tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
    if max(len(token) for token in choice_tokens) > 1:
        raise ValueError("Choice tokens must be single tokens, please use another model or change the choices.")
    choice_tokens = [token[0] for token in choice_tokens]
    return choice_tokens

def process_chunk(chunk: pd.DataFrame, model: str, gpu_id: str, **kwargs) -> List[Dict[Any, Any]]:
    """Process a chunk of data using vLLM on specified GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    logger.info(f"Running on GPU {gpu_id}")
    
    # Load vLLM config if provided
    vllm_kwargs = {}
    if offline_config := kwargs.get("offline_config"):
        vllm_kwargs = load_offline_config(offline_config)
    
    # Add model path and update with any remaining kwargs for backward compatibility
    vllm_kwargs["model"] = vllm_kwargs.get("model", kwargs.get("model", model))
    # Initialize vLLM
    llm = LLM(**vllm_kwargs)
    choice_token_ids = get_token_ids(llm=llm)
    # vLLM sampling parameters
    sampling_params = SamplingParams(
        temperature=chunk["body"].iloc[0].get("temperature", 0.7),
        max_tokens=chunk["body"].iloc[0].get("max_tokens", 1024),
        logprobs=chunk["body"].iloc[0].get("top_logprobs", None),
        top_p=chunk["body"].iloc[0].get("top_p", 0.95),
    )
    if sampling_params.logprobs:
        sampling_params.allowed_token_ids = choice_token_ids

    messages = chunk["body"].apply(lambda x: x["messages"])

    # Generate completions
    responses = llm.chat(messages.tolist(), sampling_params)
    
    # Format results
    results = [json.loads(json.dumps(response, default=lambda o: getattr(o, '__dict__', str(o)))) for response in responses]
    results = pd.Series(results, index=chunk.index, name="response").to_frame()
    results["custom_id"] = chunk["custom_id"]
    return results

def run_batch_offline(
    input_file: str,
    output_file: str | None = None,
    model: str | None = "google/Gemma-2-2B-it",
    **kwargs
) -> None:
    """
    Run batch inference using vLLM's native offline mode.
    Args:
        input_file: Input file path
        output_file: Output file path
        model: Model name or path
    """
    if not output_file:
        output_file = input_file.rsplit(".", 1)[0] + "_results.jsonl"
    
    # Step 1: Get GPU groups
    vllm_config = load_offline_config(kwargs.get("offline_config"))
    pipeline_parallel_size = vllm_config.get("pipeline_parallel_size", 1)
    tensor_parallel_size = vllm_config.get("tensor_parallel_size", 1)
    gpu_ids = kwargs.get("gpu_ids", "0")
    gpu_groups = get_gpu_groups(parallel_size=tensor_parallel_size * pipeline_parallel_size, gpu_ids=gpu_ids)
    
    if len(gpu_groups) == 0:
        raise ValueError("No enough GPUs available")
    
    # Step 2: Count total lines in input file
    total_lines = sum(1 for _ in open(input_file, 'r'))
    logger.info(f"Total samples: {total_lines}")
    # Step 3: Calculate chunksize by dividing total lines by number of GPU groups
    # Ensure at least 1 item per chunk
    chunksize = max(1, 1 + (total_lines // len(gpu_groups)))
    
    # Clear output file
    open(output_file, "w").close()

    # Process in chunks
    if len(gpu_groups) == 1:
      df = pd.read_json(input_file, lines=True)
      results = process_chunk(df ,model, "0", **kwargs)
      results.to_json(output_file, lines=True, orient="records", force_ascii=False, mode="a")
    else:
      with ProcessPoolExecutor(max_workers=len(gpu_groups)) as executor:
          futures = []
          gpu_id = 0
          
          for chunk in pd.read_json(input_file, lines=True, chunksize=chunksize):
              gpu_start_id = gpu_id % len(gpu_groups)
              futures.append(
                  executor.submit(process_chunk, chunk, model, gpu_groups[gpu_start_id], **kwargs)
              )
              gpu_id += 1
          
          # Write results as they complete
          for future in as_completed(futures):
              try:
                  results = future.result()
                  results.to_json(output_file, lines=True, orient="records", force_ascii=False, mode="a")
              except Exception as e:
                  logger.error(f"Error processing chunk: {e}")


def get_gpu_groups(parallel_size: int, gpu_ids: str) -> list[str]:
    """
    Group available GPUs based on required number per group
    
    Args:
        tensor_parallel_size: Number of GPUs needed per group
        gpu_ids: GPU ID string, e.g. "0,1,3,6,7,8"
    
    Returns:
        list[str]: List of GPU groups, each element is a string of GPU IDs for one group
        Example: ["0,1", "3,6"] or ["0", "1", "3"]
    """
    # Convert GPU ID string to list
    available_gpus = gpu_ids.split(',')
    
    # Return empty list if tensor_parallel_size is invalid
    if parallel_size <= 0:
        return []
    
    result = []
    # Calculate number of complete groups possible
    num_complete_groups = len(available_gpus) // parallel_size
    
    # Group by tensor_parallel_size
    for i in range(num_complete_groups):
        start_idx = i * parallel_size
        end_idx = start_idx + parallel_size
        group = available_gpus[start_idx:end_idx]
        result.append(','.join(group))
    
    return result
