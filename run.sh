dfile=./traindata.txt
yfile=./ydict.txt
wvfile=./wordvec.txt
maxlen=40

python train.py --data_file $dfile --y_file $yfile --maxlen $maxlen --wordvec_file $wvfile
