%% Simulated Data: Binary Choice Task
rng(42); % For reproducibility
nTrials = 100; % Number of trials

% Simulated subjective value estimates for option A minus option B
ValEstimate = randn(nTrials,1); % Normally distributed values

% Simulated choices (1 = chose A, 0 = chose B)
% Higher value should more often lead to choosing option A
trueBeta = 2; % True inverse temperature
probA = 1 ./ (1 + exp(-trueBeta * ValEstimate)); % Sigmoid choice probability
outcomeType = rand(nTrials,1) < probA; % Generate binary choices

%% Fit Logistic Regression Model
whichLink = 'logit'; % Standard link function for logistic regression
[B, dev, stats] = glmfit(ValEstimate, categorical(outcomeType), 'binomial', 'link', whichLink);

