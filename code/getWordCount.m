function count = getWordCount()

fid=fopen('train_ad.txt');
%N is the index of the ad you want
%Example here display the 2nd ad for train_ad 
%total number of training ads=20311
%total number of test ads=20307
count = zeros(20311,1);
lines = textscan(fid, '%s', 'delimiter', '\n');

for i = 1:20311
    line=lines{1}{i};
    d = textscan(line, '%s', 'delimiter', ' ');
    count(i) = size(d{1},1);
end