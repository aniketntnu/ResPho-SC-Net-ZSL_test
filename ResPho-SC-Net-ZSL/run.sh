#!/bin/bash

# resphosnet on augmented data

./PHOSCnetNor3/main.py --name ./PHOSCnetNor3/ECCV/ResNet18 --mode train --epochs 500 --stopCode 10 --flagFile ./stopFlags/flagResphoscLogs.txt --model RPnet --lr 0.0001 --phos_size 180 --phoc_size 646 --language nor --train_csv /global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/IAM/wordSpotting/IAM_train_original_cleaned.csv --train_folder /global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/IAM/wordSpotting/train/images/ --valid_csv /global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/IAM/wordSpotting/IAM_valid_cleaned.csv --valid_folder /global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/IAM/wordSpotting/valid/images --test_csv_seen /global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/IAM/wordSpotting/IAM_test_cleaned.csv --test_folder_seen /global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/IAM/wordSpotting/test/images --test_csv_unseen /global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/IAM/wordSpotting/IAM_test_cleaned.csv --test_folder_unseen /global/D1/projects/ZeroShot_Word_Recognition/E2E/allData/IAM/wordSpotting/test/images/  --batch_size 32 --pretrained_weights None

# 


#python3 main_1.py --name resphoscnet --mode train --model RPnet --epochs 1 --batch_size 32 --train_csv ./data/IAM_train.csv --train_folder /cluster/datastore/aniketag/allData/global/D1/projects/ZeroShot_Word_Recognition/Transformer_ZeroShot_Word_Recognition/joakims_work/myphosc/image_data/IAM_Data/IAM_train --valid_csv ./data/IAM_train.csv --valid_folder /cluster/datastore/aniketag/allData/global/D1/projects/ZeroShot_Word_Recognition/Transformer_ZeroShot_Word_Recognition/joakims_work/myphosc/image_data/IAM_Data/IAM_train --test_csv_seen ./data/IAM_train.csv  --test_folder_seen /cluster/datastore/aniketag/allData/global/D1/projects/ZeroShot_Word_Recognition/Transformer_ZeroShot_Word_Recognition/joakims_work/myphosc/image_data/IAM_Data/IAM_train  --test_csv_unseen ./data/IAM_train.csv --test_folder_unseen /cluster/datastore/aniketag/allData/global/D1/projects/ZeroShot_Word_Recognition/Transformer_ZeroShot_Word_Recognition/joakims_work/myphosc/image_data/IAM_Data/IAM_train >> ./logs/delMe.txt 2>&1


python3 main.py --name ./ECCV/ResNet18 --mode train --epochs 100 --stopCode 10 --flagFile ./stopFlags/flagResphoscLogs.txt --model ResNet18Phosc --lr 0.0001 --phos_size 180 --phoc_size 646 --language eng 
--train_csv /cluster/datastore/aniketag/allData/ResPhoscNetData//IAM_train_original_cleaned.csv 
--train_folder /cluster/datastore/aniketag/allData/ResPhoscNetData//train/images/ 
--valid_csv /cluster/datastore/aniketag/allData/ResPhoscNetData//IAM_valid_cleaned.csv 
--valid_folder /cluster/datastore/aniketag/allData/ResPhoscNetData//valid/images 
--test_csv_seen /cluster/datastore/aniketag/allData/ResPhoscNetData//IAM_test_cleaned.csv 
--test_folder_seen /cluster/datastore/aniketag/allData/ResPhoscNetData//test/images 
--test_csv_unseen /cluster/datastore/aniketag/allData/ResPhoscNetData//IAM_test_cleaned.csv 
--test_folder_unseen /cluster/datastore/aniketag/allData/ResPhoscNetData//test/images/  
--batch_size 32 
--pretrained_weights None
