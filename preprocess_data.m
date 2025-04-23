% preprocess_data.m
% MATLAB script for data preprocessing equivalent to the provided Python code

% 1. Read the raw CSV dump
data = readtable('data/data.csv');

% 2. Extract labels: CONS_NO (ID) & FLAG (target)
labels = data(:, {'CONS_NO','FLAG'});
writetable(labels, 'data/label.csv');

% 3. Build features matrix by dropping ID & target
features = removevars(data, {'CONS_NO','FLAG'});

% 4. Ensure number of feature columns is divisible by 7
n = width(features);
drop = mod(n, 7);
if drop > 0
    % Drop the rightmost 'drop' columns
    features = features(:, 1:(n-drop));
    % Alternatively, to pad with zeros instead of dropping:
    % padVars = array2table(zeros(height(features), drop), ...
    %     'VariableNames', strcat('PAD_', string(1:drop)));
    % features = [features padVars];
end

% 5. Save the preprocessed feature set
writetable(features, 'data/after_preprocess_data.csv');

fprintf('Saved:\n');
fprintf('  • data/label.csv             ← (CONS_NO, FLAG)\n');
fprintf('  • data/after_preprocess_data.csv ← all other columns (%d cols)\n', width(features));

% ----------------------------------------------------------------------
% Now load, impute, scale, and re-save
% ----------------------------------------------------------------------

% Load the feature data and labels
data_loaded  = readtable('data/after_preprocess_data.csv');
label_loaded = readtable('data/label.csv');

% Convert to numeric matrix for imputation & scaling
X = table2array(data_loaded);

% Impute missing values with column-wise median
medians = nanmedian(X, 1);
for j = 1:size(X,2)
    missingIdx = isnan(X(:,j));
    X(missingIdx,j) = medians(j);
end
fprintf('Imputed all random NaNs with median.\n');

% Standardize: mean=0, std=1
mu    = mean(X, 1);
sigma = std(X, 0, 1);
X_scaled = (X - mu) ./ sigma;
fprintf(' Scaled data to mean=0, std=1.\n');

% Convert back to table and save
features_scaled = array2table(X_scaled, 'VariableNames', data_loaded.Properties.VariableNames);
writetable(features_scaled, 'data/after_preprocess_data_scaled.csv');
fprintf('Saved preprocessed data to data/after_preprocess_data_scaled.csv\n');
