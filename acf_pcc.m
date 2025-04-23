% Original and tampered consumption data
consumption_o = [5.76, 16.55, 10.35, 7.38, 9.01, 5.94, 7.89, ...
                 5.09, 6.75, 7.71, 6.42, 4.59, 3.17, 7.24, ...
                 5.27, 9.25, 8.06, 7.88, 5.69, 7.81, 4.47, ...
                 3.55, 5.17, 5.18, 0, 3.07, 4.27, 3.37];

consumption_t = [0, 0, 0.19, 0.11, 0, 0.29, 0, ...
                 0, 0.02, 0, 1.14, 0.02, 0.04, 0.33, ...
                 0, 0, 0, 0.07, 0.06, 0, 0, 0, ...
                 0.7, 0, 0, 0.22, 0.1, 0.4];

% Split into 4 weeks (7 days each)
weeks = reshape(consumption_o(1:28), 7, 4)';  % 4x7 matrix

% Compute PCC matrix
pcc_matrix = zeros(4, 4);
for i = 1:4
    for j = 1:4
        r = corrcoef(weeks(i,:), weeks(j,:));
        pcc_matrix(i,j) = r(1,2);
    end
end

% Display matrix
disp('Full 4x4 Pearson Correlation Coefficient Matrix:');
disp(round(pcc_matrix, 4));

% Plot heatmap
figure;
heatmap({'Week 1','Week 2','Week 3','Week 4'}, ...
        {'Week 1','Week 2','Week 3','Week 4'}, ...
        round(pcc_matrix, 4), ...
        'Colormap', parula, ...
        'ColorbarVisible','on');
title('Heatmap of Pearson Correlation Coefficients Between Weeks');

% Plot ACF
figure;
autocorr(consumption_o, 'NumLags', 20);
title('Autocorrelation Function (ACF) of Normal Electricity Consumption');
xlabel('Lag (days)');
ylabel('Correlation');
grid on;
