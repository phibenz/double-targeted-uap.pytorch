# Fixed Params
DATASET="cifar10"
EPSILON=0.05882
SOURCE_LOSS_FN="bounded_logit_source_sink"
OTHERS_LOSS_FN="ce"
CONFIDENCE=10
BATCH_SIZE=128
LEARNING_RATE=0.005
NUM_ITERATIONS=500
WORKERS=4
NGPU=1
SUBF="one2one_cifar10"

# 0 : airplane
# 1 : automobile
# 2 : bird
# 3 : cat
# 4 : deer
# 5 : dog
# 6 : frog
# 7 : horse
# 8 : ship
# 9 : truck

# Variable Params
TARGET_NETS="resnet20"
SOURCE_CLASSES=( "2" "4" "6" "8" "9" "0" "7" "5" "5" "0")
SINK_CLASSES=( "0" "6" "3" "3" "7" "4" "5" "6" "4" "1")

for target_net in $TARGET_NETS; do
  for i in $(seq 0 9);do
    python3 train_dt_uap.py \
      --pretrained_dataset $DATASET --pretrained_arch $target_net \
      --epsilon $EPSILON \
      --source_loss $SOURCE_LOSS_FN --others_loss $OTHERS_LOSS_FN --confidence $CONFIDENCE \
      --num_iterations $NUM_ITERATIONS \
      --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE\
      --source_classes ${SOURCE_CLASSES[$i]} --sink_classes ${SINK_CLASSES[$i]} \
      --workers $WORKERS --ngpu $NGPU \
      --result_subfolder $SUBF
  done
done