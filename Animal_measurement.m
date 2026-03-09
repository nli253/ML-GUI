clc
clear all
close all
addpath('D:\deconvolution\')
pred_folder = 'D:\Animal_DTOF';
file_out='D:\Animal DTOF\Animal2_measurement';

M = load('raw 1.mat');

%%
M=ddtofs;


%M = your_matrix;   % 43368×820

rows_per_block = 104;
num_blocks = size(M,1) / rows_per_block;

for k = 1:num_blocks
    
    row_start = (k-1)*rows_per_block + 1;
    row_end   = k*rows_per_block;
    
    block = M(row_start:row_end, :);   % 104 × 820
    tmp=block(1:2:end,:)-block(2:2:end,:);

    output_file_name = fullfile(pwd, sprintf('DTOF_%d.mat', k));
    save(output_file_name,'tmp');
end