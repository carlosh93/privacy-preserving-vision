
FILE=$1

if  [ $FILE == "checkpoints" ]; then
    # shellcheck disable=SC2125
    URL="https://correouisedu-my.sharepoint.com/:u:/g/personal/jhon2208456_correo_uis_edu_co/EfRBSsN3S9VMsErqEcajUREBdzovhloqday3du4P026OLQ?e=pIeaFe&download=1"
    mkdir -p ./checkpoints
    OUT_FILE=./checkpoints/000001_nets.ckpt
    wget -N $URL -O $OUT_FILE
    
    URL="https://correouisedu-my.sharepoint.com/:u:/g/personal/jhon2208456_correo_uis_edu_co/EXgRge-famBCi891utD8OEoB0QSMp2j6GsdkYiHAawN7IQ?e=xvebcg&download=1"
    OUT_FILE=./checkpoints/000001_nets_ema.ckpt
    wget -N $URL -O $OUT_FILE
    
    URL="https://correouisedu-my.sharepoint.com/:u:/g/personal/jhon2208456_correo_uis_edu_co/EXl0zY1zM6NFmNXQrCDz54EB5-Z2NjR4qHUze4IJ2rg5Hg?e=3blo8c&download=1"
    OUT_FILE=./checkpoints/celeba_lm_mean.npz
    wget -N $URL -O $OUT_FILE
    
    URL="https://correouisedu-my.sharepoint.com/:u:/g/personal/jhon2208456_correo_uis_edu_co/EZ4QSB0VAIBDm4i9-uOEt1UB8LUbHUS-uGRyPuYlpZycTQ?e=mABZOQ&download=1"
    OUT_FILE=./checkpoints/Model_wing.pth
    wget -N $URL -O $OUT_FILE
    
    URL="https://correouisedu-my.sharepoint.com/:u:/g/personal/jhon2208456_correo_uis_edu_co/EenPLgMgMa1IrDU7o4HeSVMBLWvrUIEvEVLGtT_p96Q-cA?e=4AhnrC&download=1"
    OUT_FILE=./checkpoints/wing.ckpt
    wget -N $URL -O $OUT_FILE

    URL="https://correouisedu-my.sharepoint.com/:u:/g/personal/jhon2208456_correo_uis_edu_co/EYEiwI-tj75PgipYLu_ia30BS22dU_sZS59TKXBsGnSbfg?e=YV2IQk&download=1"
    OUT_FILE=./checkpoints/Wing_LR_16.ckpt
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "celeba-hq-dataset" ]; then
    # shellcheck disable=SC2125
    URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
    ZIP_FILE=./data/celeba_hq.zip
    mkdir -p ./data
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data
    rm $ZIP_FILE

elif  [ $FILE == "pretrained-models" ]; then
    # shellcheck disable=SC2125
    URL=https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0
    mkdir -p ./pretrained_models
    mkdir -p ./pretrained_models/Lens
    URL="https://correouisedu-my.sharepoint.com/:u:/g/personal/jhon2208456_correo_uis_edu_co/EVq0kTjK3s1Kumfihm8bOYEBi9_Lq7mrYdrCqEyEBYn4DA?e=npeyVv&download=1"
    OUT_FILE=./pretrained_models/Lens/Model_Lens.ckpt
    wget -N $URL -O $OUT_FILE

    mkdir -p ./pretrained_models/LR
    URL="https://correouisedu-my.sharepoint.com/:u:/g/personal/jhon2208456_correo_uis_edu_co/ERVy4RAJ3yNPnfXcZgfxeAgBisIzYD-VAxJ6NJLJNkjZrA?e=MJxUCE&download=1"
    OUT_FILE=./pretrained_models/LR/Model_LR.ckpt
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "raf-models" ]; then
    URL="https://correouisedu-my.sharepoint.com/:u:/g/personal/jhon2208456_correo_uis_edu_co/EaLsA1C_g6dKqTXaiDWc3sgBN9PS_r2tDGDmSUb7a4pddw?e=wdIMyw&download=1"
    ZIP_FILE=./RAF/models/
    mkdir -p ./RAF/models/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./RAF/models/
    rm $ZIP_FILE

else
    echo "Available arguments are pretrained-network-celeba-hq and celeba-hq-dataset."
    exit 1

fi
