%% t-sne

clear all
close all
clc


%%
t = readtable('data\results_int.xlsx');
data = table2array(t(:,2:end));


%%
t = readtable('data\results_entrop.xlsx');
data = table2array(t(:,2:end));


%%
t = readtable('data\results_periferie.xlsx');
data = table2array(t(:,2:end));

data(isnan(data)) = 0;

%%
t1 = readtable('data\results_int.xlsx');
t2 = readtable('data\results_entrop.xlsx');
t3 = readtable('data\results_periferie.xlsx');
data = cat(2, table2array(t1(:,2:end)), table2array(t2(:,2:end)) , table2array(t3(:,2:end)) ) ;

data(isnan(data)) = 0;

%% 3d
tsne_results = tsne(data, 'NumDimensions', 3,'Algorithm','exact','Standardize',true,'NumPCAComponents',15);

% Plot the t-SNE results in 3D
figure;
scatter3(tsne_results(:,1), tsne_results(:,2), tsne_results(:,3), 'filled');
title('t-SNE results');
xlabel('Dimension 1');
ylabel('Dimension 2');
zlabel('Dimension 3');
grid on;

hold on;
for i = 1:size(tsne_results, 1)
    text(tsne_results(i,1), tsne_results(i,2), tsne_results(i,3), num2str(i), 'FontSize', 26, 'HorizontalAlignment', 'right');
end
hold off;

%% 2D

tsne_results = tsne(data, 'NumDimensions', 2,'Algorithm','exact','Standardize',true,'NumPCAComponents',15);


% Plot the t-SNE results
figure;
scatter(tsne_results(:,1), tsne_results(:,2), 'filled');
title('t-SNE results');
xlabel('Dimension 1');
ylabel('Dimension 2');
grid on;

hold on;
for i = 1:size(tsne_results, 1)
    text(tsne_results(i,1), tsne_results(i,2), num2str(i), 'FontSize', 16, 'HorizontalAlignment', 'right');
end
hold off;