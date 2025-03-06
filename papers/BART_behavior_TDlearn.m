function [TDdata,bestAlpha,bestAlphaRisk,bhvVars] = BART_behavior_TDlearn(ptID,whichTD)
% BART_BEHAVIOR_TDLEARN analyzes BART behavior from the perspective of a
%	temporal difference learning RL algorithm. The second input argument
%	is a string indicating whetehr to do 'vanilla' TD learning or
%	'distributional' TD learning.
% 
% The function outputs a data structure with TD model variables and optimal
% alphas for reward TD models and risk TD models. 

% author: EHS20201023
% edited: RC202202


%% extra info about task beahvioral markers.
% ::blackrock data "channel" codes::
%1: trial start ::      [1 2 3 4 11 12 13 14] = [Y O R G Yc Oc Rc Gc]
%2: responded ::        [22]
%3: inflating ::        [23 24] = [start stop]
%4: banked ::           [25]
%5: popped ::           [26]
%6: outcome shown ::    [100 101] = [correct incorrect]
%7: max rt exceeded  :: [127]
%8: trial over ::		[120]

% author: EHS20181005
% 
% % patient details.
 %ptID = '202001';
 whichTD = ['vanilla'];

parentDir = ['\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\' ptID '\Data\*.nev'];
nevList = dir(parentDir);
nevFile = fullfile(nevList.folder,nevList.name);
nevFile(strfind(nevFile,'\'))='/';

% trodeLabels = ptTrodesBART(ptID);

% initializing bhv output
TDdata.patientID = ptID;
TDdata.type = whichTD;

% loading matFile
matFile = ['\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\' ptID '\Data\' ptID '.bartBHV.mat'];
load(matFile)

% load and define triggers from nevFle
NEV = openNEV(nevFile,'overwrite');
trigs = NEV.Data.SerialDigitalIO.UnparsedData;
trigTimes = NEV.Data.SerialDigitalIO.TimeStampSec;

% [20170713] I made a small error in the handle_input2.m script such that
% sometimes there is an additional 23 after the initial 23 (signifying the
% start of the inflation). The next two lines remove that second 23.
infIdx = trigs==23;
infIdx([diff(infIdx)==0; false] & infIdx==1) = 0;

% trial IDs
isCTRL = logical([data.is_control]);
balloonIDs = trigs(trigs==1 | trigs==2 | trigs==3 | trigs==11 | trigs==12 | trigs==13 | trigs==14);
% 2= pop ; 1= bank
outcomeType = trigs(sort([find(trigs==25); find(trigs==26)]))-24;

% response times
respTimes = trigTimes(trigs==25 | trigs==26);
trialStarts = trigTimes(trigs<15);
trialStarts = trialStarts(2:length(respTimes)+1);
inflateStarts = trigTimes(infIdx);
nTrials = length(respTimes);

% adjusting for numbers of trials.
balloonIDs = balloonIDs(1:nTrials);
isCTRL = isCTRL(1:nTrials);

% points vector
pointsPerTrial = diff([0 [data.score]]);

% risk colormap
cMap(2,:) = [1 0.9 0];
cMap(3,:) = [1 0.5 0];
cMap(4,:) = [1 0 0];
cMap(1,:) = [0.5 0.5 0.5];

% risk and reward colormap
rcMap(1,:) = [0.5 0.5 0.5];
rcMap(2,:) = [1 0 0];
rcMap(3,:) = [1 0.5 0];
rcMap(4,:) = [1 0.9 0];
rcMap(5,:) = rgb('lightcoral');
rcMap(6,:) = rgb('rosybrown');
rcMap(7,:) = rgb('violet');

% colormap per trial.
balloonEdgeColorMap = ones(length(balloonIDs),3)*.5;
riskRewardEdgeColorMap = zeros(length(balloonIDs),3);

% populating balloon color map
for x = 1:3
    balloonEdgeColorMap(balloonIDs==x,:) = repmat(cMap(x+1,:),sum(balloonIDs==x),1);
end

% fitting TD learning models.
if strcmp(whichTD,'vanilla')
    % Initalize variables.
    Reward(1) = pointsPerTrial(1);
    Risk(1) = 0.5;								% starting with a coin flip for risk
    ValEstimate = zeros(1,nTrials);
    RiskEstimate = zeros(1,nTrials);
    a = 0.01:0.01:1;		% alphas;
    betas = logspace(0,1,100)-.999;

    for as = 1:length(a)
        for t = 2:nTrials

            % reward varriable.
            if strcmp(data(t).result,'banked')
                Reward(t) = pointsPerTrial(t);			% outcome on current trial.
            else
                Reward(t) = 0;
            end

            % risk variable.
            if isCTRL(t)
                Risk(t) = 0;
            else
                Risk(t) = sum(outcomeType(balloonIDs==balloonIDs(t) | balloonIDs==balloonIDs(t)+10)==1)./sum(balloonIDs==balloonIDs(t) | balloonIDs==balloonIDs(t)+10);
                % current trial risk is defined as P(pop) on previously observed balloons.
            end

            % updating risk and reward PE
            RewardPE(t) = Reward(t) - ValEstimate(t-1);
   
            ValEstimate(t) = ValEstimate(t-1) + a(as)*RewardPE(t-1);

            % if t==nTrials; keyboard; end;
        end

        %         % fit logistic regression model to estimate alpha using MLE
        %         [B,~,statsB] = mnrfit(ValEstimate',categorical(outcomeType));
        %         [Brisk,~,statsRiskB] =  mnrfit(RiskEstimate',categorical(outcomeType));

        whichLink = 'logit';
        % similar but using probit instead of logit. Logit gives the same
        % answer as above.
        B = glmfit(ValEstimate',categorical(outcomeType),'binomial','link',whichLink);
        Brisk = glmfit(RiskEstimate',categorical(outcomeType),'binomial','link',whichLink);

        % saving variables in a struct for output
        TDdata.nTrials = nTrials;
        TDdata.BalloonIDs = balloonIDs;
        TDdata.Outcome = outcomeType;
        TDdata.a = a;
        TDdata.inverseTemperature(as) = abs(B(2));
        TDdata.inverseTemperatureRisk(as) = abs(Brisk(2));
        % just from the task.
        TDdata.R(as,:) = Reward;
        TDdata.perTrialRisk(as,:) = Risk;
        % variables of interest
        TDdata.V(as,:) = ValEstimate;
        TDdata.RewardPE(as,:) = RewardPE;
        TDdata.RiskPE(as,:) = RiskPE;
        TDdata.Vrisk(as,:) = RiskEstimate;

    end

%     % adjusting for inverted inverse temperature values.
%     if sign(sum(sign(TDdata.inverseTemperatureRisk(5:45))))<0
%         TDdata.inverseTemperatureRisk = -TDdata.inverseTemperatureRisk;
%     end
%     if sign(sum(sign(TDdata.inverseTemperature(5:45))))<0
%         TDdata.inverseTemperature = -TDdata.inverseTemperature;
%     end

    % figuring out the optimal learning rate.
    maxBidx = find(islocalmax(TDdata.inverseTemperature,'MaxNumExtrema',1));
    bestAlpha = a(maxBidx);

    % figuring out the optimal learning rate fro Risk
    maxBidxR = find(islocalmax(TDdata.inverseTemperatureRisk,'MaxNumExtrema',1));
    bestAlphaRisk = a(maxBidxR);

    % maybe switch this up: pick where the curve flattens...
    % fit an asymptotic function to the value expectation curves
    ft = fittype({'1','1/x'});
    VEfit = fit([1:nTrials]',TDdata.V(maxBidx,:)',ft);
    riskVEfit = fit([1:nTrials]',TDdata.Vrisk(maxBidx,:)',ft);

    %from which trial to start evaluating linear model
    trial0 = 40;
    % correalting reward and risk values for optimal learning rate
    PElm = fitlm(TDdata.RewardPE(maxBidx,trial0:end)',TDdata.RiskPE(maxBidxR,trial0:end)');	% This data appears to cluster rather than fit a line, so going to plot with balloon color
    % [20201029]:: if the clusters look neat, it may be worth looking at separability vs. impulsivity.
    Vlm = fitlm(TDdata.V(maxBidx,trial0:end)',TDdata.Vrisk(maxBidxR,trial0:end)');
    
    % saving models for optimal alphas only
    TDdata.rewardVsRiskPElm.PEmodel = PElm;
    TDdata.valueVsRiskEstimateLM.valueVsRiskModel = Vlm;

    % plotting for best alpha
    plt = true;
    if plt
        figure
        subplot(3,8,[1 2])
        plot(1:nTrials,TDdata.RewardPE(maxBidx,:))
        xlabel('trials', 'fontsize', 12)
        ylabel('reward PE', 'fontsize', 12)
        title(sprintf('patient: %s',ptID))
        axis square tight
        yl = ylim;

        subplot(3,8,[3 4])
        histogram(TDdata.RewardPE(maxBidx,:),20,'Orientation','horizontal','DisplayStyle','stairs', 'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1.5)

        axis square
        ylim(yl);

        subplot(3,8,[5 6])
        hold on
        plot(1:nTrials,TDdata.V(maxBidx,:)','k')
        plot(VEfit,1:nTrials,TDdata.V(maxBidx,:)','ko') % add marker for where the slope flattens... to show point in which mdls are calculated.
        legend off
        hold off
        xlabel('trials', 'fontsize', 12)
        ylabel('value estimate', 'fontsize', 12)
        axis square tight
        yl = ylim;

        subplot(3,8,[7 8])
        histogram(TDdata.V(maxBidx,:),20,'Orientation','horizontal','DisplayStyle','stairs','EdgeColor',[0.5 0.5 0.5], 'LineWidth',1.5)
        axis square
        ylim(yl);

        subplot(3,8,[9 10])
        plot(1:nTrials,TDdata.RiskPE(maxBidxR,:))
        xlabel('trials', 'fontsize', 12)
        ylabel('Risk PE', 'fontsize', 12)
        axis square tight
        yl = ylim;

        subplot(3,8,[11 12])
        histogram(TDdata.RiskPE(maxBidxR,:),20,'Orientation','horizontal','DisplayStyle','stairs','EdgeColor',[0.5 0.5 0.5], 'LineWidth',1.5)
        axis square
        ylim(yl);

        subplot(3,8,[13 14])
        hold on
        plot(1:nTrials,TDdata.Vrisk(maxBidxR,:),'k')
        plot(riskVEfit,1:nTrials,TDdata.Vrisk(maxBidxR,:)','ko')
        legend off
        hold off
        xlabel('trials', 'fontsize', 12)
        ylabel('risk estimate', 'fontsize', 12)
        axis square tight
        yl = ylim;

        subplot(3,8,[15 16])
        histogram(TDdata.Vrisk(maxBidxR,:),20,'Orientation','horizontal','DisplayStyle','stairs', 'EdgeColor',[0.5 0.5 0.5], 'LineWidth',1.5)
        axis square
        ylim(yl);

        subplot(3,3,7)
        hold on
        plot(TDdata.a,TDdata.inverseTemperature./max(TDdata.inverseTemperature),'-b')
        plot(TDdata.a,TDdata.inverseTemperatureRisk./max(TDdata.inverseTemperatureRisk),'--r')
        scatter([TDdata.a(maxBidx) TDdata.a(maxBidxR)],[1 1],20,[0 0 0])
        hold off
        legend('reward','risk','maxima')
        xlabel('learning rate', 'fontsize', 12)
        ylabel('max-normalized inverse temperature', 'fontsize', 12)

        subplot(3,3,8)
        hold on
        scatter(TDdata.RewardPE(maxBidx,outcomeType==1),TDdata.RiskPE(maxBidxR,outcomeType==1),8,balloonEdgeColorMap(outcomeType==1,:),'filled')
        scatter(TDdata.RewardPE(maxBidx,outcomeType==2),TDdata.RiskPE(maxBidxR,outcomeType==2),10,balloonEdgeColorMap(outcomeType==2,:))
        hold off
        xlabel('reward PE', 'fontsize', 12)
        ylabel('risk PE', 'fontsize', 12)
        title('risk and reward PEs, colored by trial balloon color')
        %title(sprintf('t(%d) = %.2f, p = %.2f, R^2 = %.2f',PElm.DFE,PElm.Coefficients{2,3},PElm.Coefficients{2,4},PElm.Rsquared.Ordinary))
        axis square tight

        subplot(3,3,9)
        plotAdded(Vlm)
        legend('off')
        xlabel('Vreward', 'fontsize', 12)
        ylabel('Vrisk', 'fontsize', 12)
        title(sprintf('t(%d) = %.2f, p = %.2f, R^2 = %.2f',Vlm.DFE,Vlm.Coefficients{2,3},Vlm.Coefficients{2,4},Vlm.Rsquared.Ordinary))
        axis square tight

        % saving figures
        halfMaximize(gcf,'page')
        saveas(gcf,['\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\simpleTDRL_optimalAlphas\' ptID '_simpleTDRL_optimalAlphas.pdf'])
        close(gcf)
    end

    % output behavioral variables
    bhvVars.balloonEdgeColorMap = balloonEdgeColorMap;
    bhvVars.outcomeType = outcomeType;

    % for fitting models with multiple learning rates.
elseif strcmp(whichTD,'distributional')

    % expectile or quantile learning rule
    whichDistRL = 'quantile';

    % fit model to each channel
    alphas = (0.1:0.1:1)';
    nAlphas = length(alphas);

    % Initalize variables.
    Reward(:,1) = pointsPerTrial(1);
    Risk(:,1) = 0.5;
    ValEstimate = zeros(1,nTrials);
    RiskEstimate = zeros(1,nTrials);

    % looping over trials.
    for as = 1:length(alphas)
        for t = 2:nTrials
            if strcmp(data(t).result,'banked')
                Reward(t) = pointsPerTrial(t);	% outcome on current trial.
            else
                Reward(t) = 0;
            end

            % risk variable.
            if isCTRL(t)
                Risk(t) = 0;
            else
                Risk(t) = sum(outcomeType(balloonIDs==balloonIDs(t) | balloonIDs==balloonIDs(t)+10)==1)./sum(balloonIDs==balloonIDs(t) | balloonIDs==balloonIDs(t)+10);
                % current trial risk is defined as P(pop) on previously observed balloons.
            end

            % update PEs
            RewardPE(t) = Reward(t) - ValEstimate(t-1);
            RiskPE(t) = Risk(t) - RiskEstimate(t-1);
            %
            %             % updating valua
            %             RiskEstimate(t) = RiskEstimate(t-1) + a(as)*RiskPE(t-1);
            %             ValEstimate(t) = ValEstimate(t-1) + a(as)*RewardPE(t-1);

            % updating values for risk.
            if strcmp(whichDistRL,'quantile')
                if RiskPE(t)>0
                    RiskEstimate(t) = RiskEstimate(t-1) + alphas(as).*1;
                elseif RiskPE(t)<=0
                    RiskEstimate(t) = RiskEstimate(t-1) + (1-alphas(as)).*-1;
                end
            elseif strcmp(whichDistRL,'expectile')
                if RiskPE(t)>0
                    RiskEstimate(t) = RiskEstimate(t-1) + alphas(as).*RiskPE(t);
                elseif RiskPE(t)<=0
                    RiskEstimate(t) = RiskEstimate(t-1) + (1-alphas(as)).*RiskPE(t);
                end
            end

            if strcmp(whichDistRL,'quantile')
                if RewardPE(t)>0
                    ValEstimate(t) = ValEstimate(t-1) + alphas(as).*1;
                elseif RewardPE(t)<=0
                    ValEstimate(t) = ValEstimate(t-1) + (1-alphas(as)).*-1;
                end
            elseif strcmp(whichDistRL,'expectile')
                if RewardPE(t)>0
                    ValEstimate(t) = ValEstimate(t-1) + alphas(as).*RewardPE(t);
                elseif RewardPE(t)<=0
                    ValEstimate(t) = ValEstimate(t-1) + (1-alphas(as)).*RewardPE(t);
                end
            end
        end

        whichLink = 'logit';
        % similar but using probit instead of logit. Logit gives the same
        % answer as above.
        B = glmfit(ValEstimate',categorical(outcomeType),'binomial','link',whichLink);
        Brisk = glmfit(RiskEstimate',categorical(outcomeType),'binomial','link',whichLink);

        % saving variables in a struct for output
        TDdata.nTrials = nTrials;
        TDdata.BalloonIDs = balloonIDs;
        TDdata.Outcome = outcomeType;
        TDdata.a = alphas;
        TDdata.inverseTemperature(as) = B(2);
        TDdata.inverseTemperatureRisk(as) = Brisk(2);
        % just from the task.
        TDdata.R(as,:) = Reward;
        TDdata.perTrialRisk(as,:) = Risk;
        % variables of interest
        TDdata.V(as,:) = ValEstimate;
        TDdata.RewardPE(as,:) = RewardPE;
        TDdata.RiskPE(as,:) = RiskPE;
        TDdata.Vrisk(as,:) = RiskEstimate;

    end
    
    % adjusting for inverted inverse temperature values.
    if sign(sum(sign(TDdata.inverseTemperatureRisk(5:25))))<0
        TDdata.inverseTemperatureRisk = -TDdata.inverseTemperatureRisk;
    end
    if sign(sum(sign(TDdata.inverseTemperature(5:25))))<0
        TDdata.inverseTemperature = -TDdata.inverseTemperature;
    end

    % figuring out the optimal learning rate.
    [~,maxBidx] = max(TDdata.inverseTemperature(1:end-10));
    bestAlpha = alphas(maxBidx);

    % figuring out the optimal learning rate fro Risk
    [~,maxBidxR] = max(TDdata.inverseTemperatureRisk(1:end-10));
    bestAlphaRisk = alphas(maxBidxR);

    % plotting for best alphas
    for xx = 1
        plt = true;
        if plt
            figure
            % plotting RPE
            subplot(3,8,[1 2])
            plot(1:nTrials,TDdata.RewardPE(maxBidx,:))
            xlabel('trials')
            ylabel('reward PE')
            title(sprintf('patient: %s',ptID))
            axis square tight
            yl = ylim;

            % plotting RPE histogram.
            subplot(3,8,[3 4])
            histogram(TDdata.RewardPE(maxBidx,:),20,'Orientation','horizontal','DisplayStyle','stairs')
            axis square
            ylim(yl);

            % plotting value estimate
            subplot(3,8,[5 6])
            plot(1:nTrials,TDdata.V(maxBidx,:))
            xlabel('trials')
            ylabel('value estimate')
            axis square tight
            yl = ylim;

            % plotting value estimate histogram
            subplot(3,8,[7 8])
            histogram(TDdata.V(maxBidx,:),20,'Orientation','horizontal','DisplayStyle','stairs')
            axis square
            ylim(yl);

            % plotting risk PE
            subplot(3,8,[9 10])
            plot(1:nTrials,TDdata.RiskPE(maxBidxR,:))
            xlabel('trials')
            ylabel('Risk PE')
            axis square tight
            yl = ylim;

            % plotting risk PE histogram
            subplot(3,8,[11 12])
            histogram(TDdata.RiskPE(maxBidxR,:),20,'Orientation','horizontal','DisplayStyle','stairs')
            axis square
            ylim(yl);

            % plotting risk estimate 
            subplot(3,8,[13 14])
            plot(1:nTrials,TDdata.Vrisk(maxBidxR,:))
            xlabel('trials')
            ylabel('risk estimate')
            axis square tight
            yl = ylim;

            % plotting risk estimate histogram
            subplot(3,8,[15 16])
            histogram(TDdata.Vrisk(maxBidxR,:),20,'Orientation','horizontal','DisplayStyle','stairs')
            axis square
            ylim(yl);

            % inverse temps and optimal alphas. 
            subplot(3,3,7)
            hold on
            plot(TDdata.a,TDdata.inverseTemperature./max(TDdata.inverseTemperature),'-b')
            plot(TDdata.a,TDdata.inverseTemperatureRisk./max(TDdata.inverseTemperatureRisk),'--r')
            scatter([TDdata.a(maxBidx) TDdata.a(maxBidxR)],[1 1],20,[0 0 0])
            hold off
            legend('reward','risk','maxima')
            xlabel('learning rate')
            ylabel('max-normalized inverse temperature')
            title(['optimal alphas | reward: ' num2str(bestAlpha) ' | risk: ' num2str(bestAlphaRisk)])

            % model variables by outcome & trial type. 
            subplot(3,3,8)
            hold on
            scatter(TDdata.RewardPE(maxBidx,outcomeType==1),TDdata.RiskPE(maxBidxR,outcomeType==1),8,balloonEdgeColorMap(outcomeType==1,:),'filled')
            scatter(TDdata.RewardPE(maxBidx,outcomeType==2),TDdata.RiskPE(maxBidxR,outcomeType==2),10,balloonEdgeColorMap(outcomeType==2,:))
            hold off
            xlabel('reward PE')
            ylabel('risk PE')
            title('risk and reward PEs, colored by trial balloon color')
            %title(sprintf('t(%d) = %.2f, p = %.2f, R^2 = %.2f',PElm.DFE,PElm.Coefficients{2,3},PElm.Coefficients{2,4},PElm.Rsquared.Ordinary))
            axis square tight

            % saving figures
            halfMaximize(gcf,'left')
            saveas(gcf,['\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\distributionalTDlearning\' ptID '_' whichDistRL '_distributionalTDlearning.pdf'])
        end
    end
end




