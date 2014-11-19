function read_ad(N)
%read in the file you want, here we use train_ad.tx
fid=fopen('train_ad.txt');
%N is the index of the ad you want
%Example here display the 2nd ad for train_ad 
%total number of training ads=20311
%total number of test ads=20307
N
line=textscan(fid,'%s', 1, 'delimiter', '\n','HeaderLines', N-1);
line{1}{1}

end