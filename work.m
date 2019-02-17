clear all; close all;

% Student number = 218886
% 218886 mod 5 = 1 => using Image_1.jpg
im = imread('Image_1.png');
A = double(im); %change from uint8 to double
symbols = 0:255;

%% Calculate histogram
imMax = max(A(:));
imMin = min(A(:));
hh = histc(A(:),symbols);%imMin:imMax);
%Plot histogram
figure;plot(symbols,hh);% plot(imMin:imMax,hh);
title('histogram of used Image');
xlabel('Gray levels');
ylabel('Number of occurrences');
xlim([0 255]);

%% Empirical entropy
pp = hh/sum(hh); %empirical probability
ind = find(pp>0);
ExistingSymbols = symbols(ind);
SymbProb = pp(ind);
H = -sum(SymbProb.*log2(SymbProb)); %entropy
disp(['Empirical entropy for the image: ', num2str(H)]);

%% Take only the rows and columns between 50-249
B = A(50:249,50:249);
imwrite(uint8(B),'MyImage 1.png','png');

%% % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % Martucci predictor (Median Adaptive Predictor)% % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

[rows,cols] = size(B); %image dimensions
eMtx = zeros(rows,cols); %initialize error matrix
for i = 2:rows %for each row
    for j = 2:cols %for each column
        y_n = B(i-1,j); %y_north estimate
        y_w = B(i,j-1); %y_west estimate
        y_nw = B(i-1,j-1); %y_northwest
        y_map = median([y_n,y_w,y_n+y_w-y_nw]); %MAP estimate for B(i,j)
        eMtx(i,j) = B(i,j)-y_map; %prediction error matrix
        eVect = eMtx(:); %prediction error vector (concatenated each column)
    end
end

% Plot the prediction error matrix E using the imshow MATLAB function. Shift all negative
% integer error samples to positive range for better visualization
imshow(eMtx)

%% % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % Golomb-Rice coding % % % % % %  % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% So that we can use Golob-Rice coding, we convert signed values into
% unsigned values (following the examples of course webpage)
eVect(eVect>=0) = 2*eVect(eVect>=0);
eVect(eVect<0) = -2*eVect(eVect<0)+1;

L = (50:50:2000); %block length vector 50,100,150,...,2000
best_p_vector = [];
comp_size = 4*8+4*8;

for block_indx = 1:length(L) %for each block length
   block_cnt = floor(length(eVect)/L(block_indx)); %how many blocks with specific block length
   for indx = 1:block_cnt
       block = eVect((1+(indx-1)*L(block_indx)):(indx*L(block_indx))); %take block of data
       [best_size,best_p] = GR_estimation(block); %estimate variable p
       comp_size = comp_size+4+best_size; %save size
       best_p_vector = [best_p_vector; best_p]; %save best variable p
   end
end
stem(best_p_vector);

comp_size_bytes = ceil(comp_size/8);
disp(['Compressed file size: ' num2str(comp_size_bytes) ' bytes'])


%% 
function [best_size,best_p] = GR_estimation(y)
best_size = 1e10; %initialize
best_p = -1;
len = length(y);
for p = 0:15 %Going through different number of bits
    size = sum(floor(y/2^p))+1*len+p*len; %Codeword length for coding an integer
    if size < best_size
        best_size = size;
        best_p = p;
    end
end
end






