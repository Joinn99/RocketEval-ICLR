# This script shows how to run tasks in RocketEval.

# Set default parameters for tasks
DATASET=mt-bench
GENERATOR=Qwen2.5-7B-Instruct       # Name of the checklist generator model
JUDGE=Qwen2.5-0.5B-Instruct         # Name of the judge model
LABELER=gpt-4o                      # Name of the labeler model

# Set API key and base URL
export OPENAI_API_KEY=<API_KEY>
export OPENAI_BASE_URL=<URL>

# -------------------------------------FULL TASKS---------------------------------------

# Running Full Tasks (API mode, with checklist generation)
python src/run.py --dataset ${DATASET} --generator ${GENERATOR} --judge ${JUDGE} \
    --mode api \
    --train_test \
    --gen_checklist

# Running Full Tasks (Offline mode, without checklist generation)
python src/run.py --dataset ${DATASET} --generator ${GENERATOR} --judge ${JUDGE} \
    --mode offline \
    --train_test \
    --gpu_ids "0"

# ----------------------------------INDIVIDUAL TASKS------------------------------------

# Running Checklist Generation Task (API mode, using instant API with parallel size 32)
python src/run_task.py checklist --dataset ${DATASET} --generator ${GENERATOR} \
    --mode api \
    --instant_api \
    --api_parallel_size 32

# Running Checklist Grading Task (Offline mode with data parallelism, specifying offline config file)
python src/run_task.py judgment --dataset ${DATASET} --judge ${JUDGE} \
    --train_test \
    --mode offline \
    --offline_config config/offline/colab.yaml \
    --gpu_ids "0,1,2,3"

# Running Score Prediction Task
python src/run_task.py score --dataset ${DATASET} --judge ${JUDGE} --labeler ${LABELER} \
    --train_test

# Running Ranking Production Task
python src/run_task.py ranking --dataset ${DATASET} --judge ${JUDGE} \
    --train_test
