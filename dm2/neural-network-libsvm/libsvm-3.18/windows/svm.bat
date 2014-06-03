F:
cd F:\大三资料\SRTP\libsvm-3.18\libsvm-3.18\windows
svm-scale.exe -l -1 -u 1 -s range1 train_data.txt > train.scale
svm-scale.exe -r range1 test_data.txt > test.scale
python gridregression.py -log2c -10,10,1 -log2g -10,10,1 -log2p -10,10,1 -v 10 -gnuplot C:\Program Files\gnuplot\bin\gnuplot.exe  -s 3 -t 2 -h 0 train.scale > parameter