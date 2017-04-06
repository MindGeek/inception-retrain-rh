timestamp=$(date '+%m-%d.%H_%M')

python ./retrain.py \
--bottleneck_dir=./tf_files/bottlenecks \
--how_many_training_steps 800 \
--train_batch_size 128 \
--learning_rate 0.01 \
--model_dir=./models \
--output_graph=./tf_files/retrained_graph.pb \
--output_labels=./tf_files/retrained_labels.txt \
--sample_dir=./data/training \
--final_model_path=./models/rh_model.${timestamp} \
--summaries_dir ./retrain_summaries/${timestamp}
