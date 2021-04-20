from Evaluator import MCPE
from simpleMC import MChain
import numpy
import matplotlib.pyplot as plt
from numpy import math, Inf, reshape
from scipy import linalg
from Evaluator import MCPE
from radialBasis import radialBasisFunctions
import csv
import argparse
import os
from scipy.stats import sem




class experiment():
    def __init__(self, aggregationFactor, stateSpace, epsilon, delta, lambdaClass, numRounds, batchSize,
                 policy="uniform", batch_gen_param="N"):
        self.__Phi = self.featureProducer(aggregationFactor, stateSpace)
        self.__epsilon = epsilon
        self.__delta = delta
        self.__lambdaCalss = lambdaClass
        self.__numrounds = numRounds
        self.__batchSize = batchSize
        self.__policy = policy
        self.__stateSpace = stateSpace
        self.__batch_gen_param_trigger = batch_gen_param

    def getBatchSize(self):
        return self.__batchSize

    def radialfeature(self, stateSpace):
        myExpParams = expParameters()
        myradialBasis = radialBasisFunctions(stateSpace, myExpParams.means, myExpParams.sigmas)
        return myradialBasis.phiGen()

    def getPhi(self):
        return self.__Phi

    def getPolicy(self):
        return self.__policy

    def lambdaExperiment_SA_LSL(self, mdp, n_subsamples, batchsize, max_trajectory_length, regCoefs, pow_exp, s,
                                epsilon, delta, Phi, distUB=10):
        myMCPE = MCPE(mdp, self.__Phi, self.__policy)
        V = myMCPE.realV(mdp)
        dim = len(numpy.mat(self.__Phi).T)
        rho = mdp.startStateDistribution()
        maxR = mdp.getMaxReward()
        res = []
        for k in range(self.__numrounds):
            if (self.__batch_gen_param_trigger == "Y"):
                S = myMCPE.batchGen(mdp, max_trajectory_length, batchsize, mdp.getGamma(), self.__policy, rho)
            else:
                S = myMCPE.batchCutoff("huge_batch.txt", batchsize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            errls = []
            ridgeParams_orig = myMCPE.computeLambdas(mdp, self.__Phi, regCoefs, int(batchsize / n_subsamples), pow_exp)
            ridgeParams = []
            for l in range(len(ridgeParams_orig)):
                ridgeParams.append(ridgeParams_orig[l][0])
            for i in range(len(ridgeParams)):
                # tL=myMCPE.LSL(FVMC[2], mdp, self.__Phi, ridgeParams[i], len(S),FVMC[1])
                DPSA, SA = myMCPE.lsl_sub_sample_aggregate(S, s, n_subsamples, mdp, Phi, ridgeParams[i], pow_exp,
                                                           batchsize, epsilon, delta, distUB)
                DPSA = reshape(DPSA, (len(DPSA), 1))
                SA = reshape(SA, (len(SA), 1))
                diff_V_SA = myMCPE.weighted_dif_L2_norm(mdp, V, SA)
                diff_V_DPSA = myMCPE.weighted_dif_L2_norm(mdp, V, DPSA)
                errls.append([ridgeParams_orig[i][1], diff_V_SA, diff_V_DPSA])
            res.append(errls)
        return res

    def lambdaExperiment_LSL(self, mdp, batchSize, maxTrajectoryLenghth, regCoefs, pow_exp):
        myMCPE = MCPE(mdp, self.__Phi, self.__policy)
        V = myMCPE.realV(mdp)
        dim = len(numpy.mat(self.__Phi).T)
        rho = mdp.startStateDistribution()
        maxR = mdp.getMaxReward()
        res = []
        for k in range(self.__numrounds):
            if (self.__batch_gen_param_trigger == "Y"):
                S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)
            else:
                S = myMCPE.batchCutoff("huge_batch.txt", batchSize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            errls = []
            ridgeParams_orig = myMCPE.computeLambdas(mdp, self.__Phi, regCoefs, self.__batchSize, pow_exp)
            ridgeParams = []
            for l in range(len(ridgeParams_orig)):
                ridgeParams.append(ridgeParams_orig[l][0])
            for i in range(len(ridgeParams)):
                tL = myMCPE.LSL(FVMC[2], mdp, self.__Phi, ridgeParams[i], len(S), FVMC[1])
                VL = self.__Phi * tL
                dpLSL = myMCPE.DPLSL(tL, FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon, self.__delta,
                                     ridgeParams[i], len(S), rho, self.__policy)
                temp5 = reshape(dpLSL[0], (len(dpLSL[0]), 1))
                dpVL = self.__Phi * temp5
                diff_V_VL = myMCPE.weighted_dif_L2_norm(mdp, V, VL)
                diff_V_dpVL = myMCPE.weighted_dif_L2_norm(mdp, V, dpVL)
                errls.append([ridgeParams_orig[i][1], diff_V_VL, diff_V_dpVL])

            res.append(errls)
        return res

    def lambdaExperiment_GS_LSL(self, myMCPE, mdp, batchSize, maxTrajectoryLenghth, regCoefs, pow_exps):
        myMCPE = myMCPE
        V = myMCPE.realV(mdp)
        dim = len(self.__Phi)
        rho = mdp.startStateDistribution()
        maxR = mdp.getMaxReward()
        res = []
        for k in range(self.__numrounds):
            if (self.__batch_gen_param_trigger == "Y"):
                S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)
            else:
                S = myMCPE.batchCutoff("huge_batch.txt", batchSize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            errls = []
            ridgeParams = myMCPE.computeLambdas(mdp, self.__Phi, regCoefs, self.__batchSize, pow_exps)
            for i in range(len(ridgeParams)):
                tL = myMCPE.LSL(FVMC[2], mdp, self.__Phi, ridgeParams[i], len(S))
                VL = self.__Phi * tL
                dpLSL_smoothed = myMCPE.DPLSL(FVMC[2], FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon,
                                              self.__delta, ridgeParams[i], len(S), rho, self.__policy)
                dpLSL_GS = myMCPE.GS_based_DPLSL(FVMC[2], FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon,
                                                 self.__delta, ridgeParams[i], len(S), rho, self.__policy)
                temp5 = reshape(dpLSL_smoothed[0], (len(dpLSL_smoothed[0]), 1))
                temp6 = reshape(dpLSL_GS[0], (len(dpLSL_GS[0]), 1))
                dpVL_smoothed = self.__Phi * temp5
                dpVL_GS = self.__Phi * temp6
                diff_V_VL = myMCPE.weighted_dif_L2_norm(mdp, V, VL)
                diff_V_dpVL_smoothed = myMCPE.weighted_dif_L2_norm(mdp, V, dpVL_smoothed)
                diff_V_dpVL_GS = myMCPE.weighted_dif_L2_norm(mdp, V, dpVL_GS)
                errls.append([ridgeParams[i], diff_V_VL, diff_V_dpVL_smoothed, diff_V_dpVL_GS])

            res.append([myMCPE.weighted_dif_L2_norm(mdp, V, FVMC[2]), errls])
        return res

    # This is the experiment we run to compare
    def newGS_LSL_experiments(self, batchSize, mdp, maxTrajectoryLenghth, regCoef, pow_exp):
        myMCPE = MCPE(mdp, self.__Phi, self.__policy)
        V = myMCPE.realV(mdp)
        V = numpy.reshape(V, (len(V), 1))
        rho = mdp.startStateDistribution()
        ridgeParam = myMCPE.computeLambdas(mdp, self.__Phi, regCoef, batchSize, pow_exp)
        err_new_lsl = []
        for k in range(self.__numrounds):
            if (self.__batch_gen_param_trigger == "Y"):
                S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)
            else:
                S = myMCPE.batchCutoff("huge_batch.txt", batchSize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            tL = myMCPE.LSL(FVMC[2], mdp, self.__Phi, ridgeParam[0], batchSize, FVMC[1])
            VL = self.__Phi * tL
            dpLSL = myMCPE.GS_based_DPLSL(FVMC[2], FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon,
                                          self.__delta, ridgeParam[0], batchSize, rho, self.__policy)
            dpLSL_smoothed = myMCPE.DPLSL(FVMC[2], FVMC[1], mdp, self.__Phi, mdp.getGamma(), self.__epsilon,
                                          self.__delta, ridgeParam[0], batchSize, rho, self.__policy)
            # dpLSL=reshape(dpLSL[0], (len(dpLSL[0]),1))
            # dpLSL_smoothed=reshape(dpLSL_smoothed[0], (len(dpLSL_smoothed[0]),1))
            dpVL_GS = self.__Phi * dpLSL[0]
            dpVL_smoothed = self.__Phi * dpLSL_smoothed[0]
            diff_V_VL = myMCPE.weighted_dif_L2_norm(mdp, V, VL)
            diff_V_dpVLGS = myMCPE.weighted_dif_L2_norm(mdp, V, dpVL_GS)
            diff_V_dpVL_smoothed = myMCPE.weighted_dif_L2_norm(mdp, V, dpVL_smoothed)
            err_new_lsl.append([ridgeParam[0], diff_V_VL, diff_V_dpVLGS, diff_V_dpVL_smoothed])
        return err_new_lsl

    def TransitionFunction(self, sourceState, destState):
        if self.__policy is "uniform":
            if destState == sourceState and sourceState != len(self.__stateSpace) - 1:
                return 1.0 / 2
            if sourceState == destState - 1 and sourceState != len(self.__stateSpace) - 1:
                return 1.0 / 2
            if sourceState == len(self.__stateSpace) - 1:
                if sourceState == destState:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0

    def featureProducer(self, aggregationFactor, stateSpace):
        if aggregationFactor == 1:
            return numpy.matrix(numpy.identity(len(stateSpace)), copy=False)
        else:
            aggregatedDim = int(len(stateSpace) / aggregationFactor)
            aggFeatureMatrix = [[0 for col in range(len(stateSpace))] for row in range(int(aggregatedDim))]
            k = 0
            for i in range(aggregatedDim):
                for j in range(len(stateSpace)):
                    if (j - i) - k == 1 or j - i - k == 0:
                        aggFeatureMatrix[i][j] = 1
                    else:
                        aggFeatureMatrix[i][j] = 0
                k = k + 1
            featureMatrix = numpy.reshape(aggFeatureMatrix, (aggregatedDim, len(stateSpace)))
        return featureMatrix.T

    def lsw_sub_sample_aggregate_experiment(self, mdp, batchSize, maxTrajectoryLenghth, number_of_sub_samples,
                                            epsilon_star,
                                            delta_star, delta_prime, Phi, subSampleSize, result_path, args):
        myMCPE = MCPE(mdp, self.__Phi, self.__policy)
        V = myMCPE.realV(mdp)
        rho = mdp.startStateDistribution()

        epsilon = math.log(0.5 + math.sqrt(0.25 + (batchSize * epsilon_star) / (
                    subSampleSize * (math.sqrt(8 * number_of_sub_samples * math.log(1 / delta_prime))))))
        delta = (batchSize * (delta_star - delta_prime) / (subSampleSize * number_of_sub_samples)) * \
                (1 / (0.5 + math.sqrt(0.25 + (batchSize * epsilon_star) / (
                        subSampleSize * (math.sqrt(8 * number_of_sub_samples * math.log(1 / delta_prime)))))))

        aggregated_lsw = []
        lsw_aggregated = []
        temFVMC = []
        DPLSW_result = []
        tempMCPE = [0, 0]
        for k in range(args.num_rounds):
            print(f"round {k} has just started")
            if (self.__batch_gen_param_trigger == "Y"):
                sample_batch = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)
            else:
                sample_batch = myMCPE.batchCutoff("newBatch.txt", batchSize)
                print(f"Batch of size {batchSize} is generated")
            first_visit_monte_carlo = myMCPE.FVMCPE(mdp, self.__Phi, sample_batch)
            DPLSW_result.append(numpy.mat(Phi) * numpy.mat(
                myMCPE.DPLSW(first_visit_monte_carlo[2], first_visit_monte_carlo[1], mdp, self.__Phi, mdp.getGamma(),
                             epsilon_star, delta_star, batchSize)[0]).T)
            tempMCPE = myMCPE.lsw_sub_sample_aggregate(sample_batch, number_of_sub_samples, mdp, self.getPhi(), epsilon,
                                                       delta, epsilon_star, delta_star, subSampleSize)
            aggregated_lsw.append(tempMCPE[0])
            lsw_aggregated.append(tempMCPE[1])
            temFVMC.append(numpy.mat(Phi) * numpy.mat(first_visit_monte_carlo[0]))
            print(f"round {k} has just finished")

        return [aggregated_lsw, lsw_aggregated, temFVMC, V, DPLSW_result]

    def compute_epsilon_tilde(self, batch_size, sub_sample_size, delta_prime, delta_star, epsilon_star,
                                number_of_sub_samples):
        epsilon_tilde = math.log(0.5 + math.sqrt(0.25 + (batch_size * epsilon_star) / (
                sub_sample_size * (math.sqrt(8 * number_of_sub_samples * math.log(1 / delta_prime))))))
        return epsilon_tilde

    def compute_delta_tilde(self, batch_size, sub_sample_size, delta_prime, delta_star, epsilon_star,
                                number_of_sub_samples):
        delta_tilde = (batch_size * (delta_star - delta_prime) / (sub_sample_size * number_of_sub_samples)) * \
                (1.0 / (0.5 + math.sqrt(0.25 + (batch_size * epsilon_star) / (
                        sub_sample_size * (math.sqrt(8 * number_of_sub_samples * math.log(1.0 / delta_prime)))))))
        return delta_tilde

    def calculate_utility_sub_sample_size(self, epsilon, num_sub_samples, batch_size):

        min = 1000

        final_sub_sample_size = 0
        final_num_sub_sample = 0
        min_num_sub_samples = math.floor(math.pow(batch_size, 1.01))
        max_num_sub_samples = math.floor(math.pow(batch_size, 1.99))
        min_sub_sample_size = math.floor(math.pow(batch_size, 0.09))
        max_sub_sample_size = math.floor(math.pow(batch_size, 0.99))
        for num_sub_samples in range(min_num_sub_samples, max_num_sub_samples):
            for sub_sample_size in range(min_sub_sample_size, max_sub_sample_size):
                temp = (math.log((sub_sample_size * num_sub_samples)/batch_size, 2.0))/(sub_sample_size*sub_sample_size *
                                                                                       num_sub_samples*epsilon*epsilon) \
                       + 1.0/(sub_sample_size*num_sub_samples*num_sub_samples)
                if temp < min and temp >0:
                    min = temp
                    final_sub_sample_size = sub_sample_size
                    final_num_sub_sample = num_sub_samples
        return final_sub_sample_size, final_num_sub_sample

    def compute_parameters(self, batch_size, delta_prime, delta_star, epsilon_star):

        final_sub_sample_size = 0
        final_num_sub_sample = 0

        min_num_sub_samples = math.floor(math.pow(batch_size, 1.01))
        max_num_sub_samples = math.floor(math.pow(batch_size, 1.99))
        min_sub_sample_size = math.floor(math.pow(batch_size, 0.09))
        max_sub_sample_size = math.floor(math.pow(batch_size, 0.99))

        top_star = math.log(1.0 / delta_star) * math.log(1.25/delta_star)
        bot_star = math.pow(batch_size, 2) * math.pow(epsilon_star, 2)
        fraction_star = top_star / bot_star

        for num_sub_samples in range(min_num_sub_samples, max_num_sub_samples):
            for sub_sample_size in range(min_sub_sample_size, max_sub_sample_size):
                delta_tilde = self.compute_delta_tilde(batch_size, sub_sample_size, delta_prime, delta_star,
                                                       epsilon_star, num_sub_samples)
                epsilon_tilde = self.compute_epsilon_tilde(batch_size, sub_sample_size, delta_prime, delta_star,
                                                           epsilon_star,
                                                           num_sub_samples)
                top_tilde = math.log(1.0 / delta_tilde) * math.log(1.25 / delta_tilde)
                bot_tilde = num_sub_samples * math.pow(sub_sample_size, 2) * math.pow(epsilon_tilde, 2)
                fraction_tilde = top_tilde / bot_tilde
                temp = (fraction_tilde + 1.0/(sub_sample_size* math.pow(num_sub_samples, 2)))
                print(temp - fraction_star)
                if temp < fraction_star:
                    final_sub_sample_size = num_sub_samples
                    final_num_sub_sample = num_sub_samples
        return final_sub_sample_size, final_num_sub_sample

    def compute_num_sub_samples(self, batch_size, sub_sample_size, delta_prime, delta_star, epsilon_star):
        candidates = []
        difs = []
        min = 1000
        for number_of_sub_samples in range(1, int(math.pow(batch_size, 2))):
            sub_sample_size = math.floor(batch_size/math.pow(number_of_sub_samples, 1))
            delta_tilde = self.compute_delta_tilde(batch_size, sub_sample_size, delta_prime, delta_star,
                                                                epsilon_star, number_of_sub_samples)
            epsilon_tilde = self.compute_epsilon_tilde(batch_size, sub_sample_size, delta_prime, delta_star,
                                                                                              epsilon_star,
                                                                                              number_of_sub_samples)

            top_tilde = math.log(1.0 / delta_tilde) #* math.log(1.25 / delta_tilde)
            bot_tilde = number_of_sub_samples * math.pow(sub_sample_size, 2) * math.pow(epsilon_tilde,2)
            top_star = math.log(1.0/delta_star) #* math.log(1.25/delta_star)
            bot_star = math.pow(batch_size, 2) * math.pow(epsilon_star, 2)
            fraction_tilde = top_tilde/bot_tilde
            fraction_star = top_star/bot_star
            if (fraction_tilde + 1.0/(sub_sample_size* math.pow(number_of_sub_samples, 2))) < fraction_star:
                candidates.append(number_of_sub_samples)
                difs.append(top_star/bot_star - top_tilde/bot_tilde - 1.0/(number_of_sub_samples *
                                                                           math.pow(sub_sample_size, 2)))
        final = candidates[difs.index(min(difs))]
        return final

    def lsl_sub_sample_aggregate_experiment(self, mdp, regCoef, batchSize, pow_exp, maxTrajectoryLenghth,
                                            numberOfsubSamples, num_sub_sample_rooted, epsilon_star, delta_star,
                                            delta_prime, Phi, distUB, subSampleSize, args):

        epsilon = math.log(0.5 + math.sqrt(0.25 + (batchSize * epsilon_star) / (
                subSampleSize * (math.sqrt(8 * numberOfsubSamples * math.log(1 / delta_prime))))))
        delta = (batchSize * (delta_star - delta_prime) / (subSampleSize * numberOfsubSamples)) * \
                (1 / (0.5 + math.sqrt(0.25 + (batchSize * epsilon_star) / (
                        subSampleSize * (math.sqrt(8 * numberOfsubSamples * math.log(1 / delta_prime)))))))

        myMCPE = MCPE(mdp, self.__Phi, self.__policy)
        V = myMCPE.realV(mdp)
        rho = mdp.startStateDistribution()
        # ridgeParam=myMCPE.computeLambdas(mdp, self.__Phi, [regCoef], batchSize, pow_exp[0])
        results_dp_aggregated_subsamples = []
        results_aggregated_dp = []
        temLSL = []
        tempDPLSL = []
        for k in range(args.num_rounds):
            print(f"round {k} has just started")
            if ( self.__batch_gen_param_trigger == "Y" ):
                sampled_batch = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)
            else:
                sampled_batch = myMCPE.batchCutoff("newBatch.txt", batchSize)
                print(f"Batch of size {batchSize} is generated")
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, sampled_batch)
            # ridgeParam=myMCPE.computeLambdas(mdp, self.__Phi, regCoef, len(S), pow_exp)
            #lsl_reidge = 10000 * math.pow(len(sampled_batch), 0.4)
            lsl_reidge = 4 * math.pow(len(sampled_batch), 0.5)
            LSL_result = myMCPE.LSL(FVMC[2], mdp, self.__Phi, lsl_reidge, len(sampled_batch), FVMC[1])
            DPLSL_result = myMCPE.DPLSL(LSL_result, FVMC[1], mdp, self.__Phi, mdp.getGamma(), epsilon, delta, lsl_reidge
                                        , len(sampled_batch), rho)[0]
            # print('LSL Norm: '+str(numpy.linalg.norm(LSL_result)))
            # print('DPLSL Norm: '+str(numpy.linalg.norm(DPLSL_result)))
            sub_sampled_lsl_vector, sub_sampled_dplsl_vector = myMCPE.lsl_sub_sample_aggregate(sampled_batch,
                                                                                               num_sub_sample_rooted,
                                                                                               numberOfsubSamples, mdp,
                                                                                               self.getPhi(), epsilon,
                                                                                               delta, epsilon_star,
                                                                                               delta_star, rho,
                                                                                               subSampleSize)
            lsl_reidge_sub_sampled = 4 * math.pow(subSampleSize, 0.5)
            dplsl_sub_sampled_vector = myMCPE.DPLSL(sub_sampled_lsl_vector, FVMC[1], mdp, self.__Phi, mdp.getGamma(),
                                                    epsilon_star, delta_star, lsl_reidge_sub_sampled, len(sampled_batch), rho)[0]
            results_dp_aggregated_subsamples.append(numpy.mat(Phi)*numpy.mat(dplsl_sub_sampled_vector))
            results_aggregated_dp.append(sub_sampled_dplsl_vector)
            temLSL.append(numpy.mat(Phi) * numpy.mat(LSL_result))
            tempDPLSL.append(numpy.mat(Phi) * numpy.mat(DPLSL_result))
            print(f"round {k} has just finished")
        return [results_dp_aggregated_subsamples, results_aggregated_dp, temLSL, V, tempDPLSL]

    def rewardfunc(self, destState, goalstates, maxReward):
        if destState in goalstates:
            return maxReward
        else:
            return 0


def run_sub_sample_size_experiment(exps, args):
    results = []
    batch_size = args.batch_size
    sub_sample_size = math.floor(math.pow(batch_size, 0.5))
    num_sub_samples = math.floor(math.pow(batch_size, 0.5))
    # for epsilon in args.values_epsilon:
    #     results.append(exps.compute_num_sub_samples(batch_size, sub_sample_size, args.delta_prime, args.delta,
    #                                                 epsilon))

    for epsilon in args.values_epsilon:
        results.append(exps.compute_parameters(batch_size, args.delta_prime, args.delta, epsilon))

    ax = plt.gca()
    ax.set_prop_cycle(color=['red'])
    ax.plot(results)

    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.ylabel('(log10) epsilon')
    plt.xlabel('sub-sample size')
    plt.gcf().subplots_adjust(bottom=0.18, left=0.18)
    plt.tick_params(labelsize=18)
    ax.legend(["True vs. DP-LSW", "True vs. Aggregation on DP-LSWs", "True vs. DP-LSW on aggregations"], loc='lower left')
    plt.savefig(result_path + '/' + str(args.delta_prime) + '_' + str(args.delta) + '.png')
    plt.figure(1)
    plt.show()


def run_lambda_experiment_lsl(experimentList, args, myMDP):
    i = 0
    expResults = []
    for i in range(len(args.experiment_batch_lenghts)):
        expResults.append(experimentList[i].lambdaExperiment_LSL(myMDP, args.experiment_batch_lenghts[i],
                                                                 args.maxTrajLength, args.reg_coefs,
                                                                 args.pow_exp))
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    # ax.set_xscale('log')
    ax.set_yscale('log')
    num_reidge_params = len(args.reg_coefs) * len(args.pow_exp)

    # Real_vs_LSL_list=numpy.zeros(len(num_reidge_params))
    # Real_vs_DPLSL_list=numpy.zeros(len(num_reidge_params))
    expReal_vs_LS = numpy.zeros((len(args.experiment_batch_lenghts), num_reidge_params))
    expReal_vs_DPLSL = numpy.zeros((len(args.experiment_batch_lenghts), num_reidge_params))
    reidgeParamLsit = []
    i = 0
    num_reidge_params = len(args.reg_coefs) * len(args.pow_exp)
    for i in range(len(args.experiment_batch_lenghts)):
        Real_vs_LSL_list = numpy.zeros((num_reidge_params))
        Real_vs_DPLSL_list = numpy.zeros((num_reidge_params))
        for j in range(args.numRounds):
            tempLSL = []
            tempDPLSL = []
            reidgeParamLsit = []
            for k in range(num_reidge_params):
                reidgeParamLsit.append(expResults[i][j][k][0])
                tempLSL.append(expResults[i][j][k][1])
                tempDPLSL.append(expResults[i][j][k][2])
            Real_vs_LSL_list += numpy.ravel((1 / args.numRounds) * numpy.mat(tempLSL))
            Real_vs_DPLSL_list += numpy.ravel((1 / args.numRounds) * numpy.mat(tempDPLSL))
        expReal_vs_LS[i] = Real_vs_LSL_list
        expReal_vs_DPLSL[i] = Real_vs_DPLSL_list

    # ax.plot(numpy.ravel(expReal_vs_LS[0]))
    # ax.plot(numpy.ravel(expReal_vs_DPLSL[0]))
    ax.plot(numpy.ravel(expReal_vs_LS[0]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[0]))
    # print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[1])-numpy.ravel(expReal_vs_DPLSL[1]))])
    ax.plot(numpy.ravel(expReal_vs_LS[1]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[1]))
    # print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[2])-numpy.ravel(expReal_vs_DPLSL[3]))])
    ax.plot(numpy.ravel(expReal_vs_LS[2]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[2]))
    # print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[3])-numpy.ravel(expReal_vs_DPLSL[3]))])
    print(reidgeParamLsit[28])
    plt.show()


def run_lambdaExperiment_GS_LSL(myMCPE, experimentList, myMDP_Params, myExp_Params, myMDP):
    i = 0

    meta_exponenet_test_Reuslts = []
    for m in range(len(myExp_Params.pow_exp)):
        expResults = []
        for i in range(len(myExp_Params.experiment_batch_lenghts)):
            expResults.append(
                experimentList[i].lambdaExperiment_GS_LSL(myMCPE, myMDP, myExp_Params.experiment_batch_lenghts[i],
                                                          myExp_Params.maxTrajLength, myExp_Params.reg_coefs,
                                                          myExp_Params.pow_exp[m]))
        meta_exponenet_test_Reuslts.append(expResults)
    ax = plt.gca()
    # ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    color_cycle = ['b', 'r', 'g', 'c', 'k', 'y', 'm']
    ax.set_xscale('log')
    ax.set_yscale('log')
    realV_vs_FVMC = numpy.zeros(len(myExp_Params.reg_coefs))
    diff_V_dpVL_smoothed = numpy.zeros(len(myExp_Params.reg_coefs))
    diff_V_dpVL_GS = numpy.zeros(len(myExp_Params.reg_coefs))
    regCoefVals = numpy.zeros(len(myExp_Params.reg_coefs))
    i = 0
    # for i in range(len(args.experiment_batch_lenghts)):
    m = 0
    for m in range(len(myExp_Params.pow_exp)):
        diff_V_dpVL_smoothed = numpy.zeros(len(myExp_Params.reg_coefs))
        diff_V_dpVL_GS = numpy.zeros(len(myExp_Params.reg_coefs))
        regCoefVals = numpy.zeros(len(myExp_Params.reg_coefs))
        for k in range(len(myExp_Params.experiment_batch_lenghts)):
            for i in range(len(myExp_Params.reg_coefs)):
                for j in range(myExp_Params.numRounds):
                    diff_V_dpVL_smoothed[i] += (meta_exponenet_test_Reuslts[m][k][j][1][i][2] / myExp_Params.numRounds)
                    diff_V_dpVL_GS[i] += (meta_exponenet_test_Reuslts[m][k][j][1][i][3] / myExp_Params.numRounds)
                    regCoefVals[i] = meta_exponenet_test_Reuslts[m][k][0][1][i][0]
            ax.plot(regCoefVals, diff_V_dpVL_smoothed, color=color_cycle[k])
            ax.plot(regCoefVals, diff_V_dpVL_GS, 'r--', color=color_cycle[k])
            ax.legend(
                ["diff_V_dpVL_smoothed, " + " m: " + str(myExp_Params.experiment_batch_lenghts[k]), " diff_V_dpVL_GS"],
                loc=1)
            min_dif = numpy.linalg.norm(diff_V_dpVL_smoothed - diff_V_dpVL_GS, -numpy.Inf)
            print(str(min_dif) + "batch size= " + str(myExp_Params.experiment_batch_lenghts[k]) + " c= " + str(
                myExp_Params.reg_coefs[i]) + " exponent= " + str(myExp_Params.pow_exp[m]))
        plt.xlabel("exponent= " + str(myExp_Params.pow_exp[m]))
        plt.show()


def run_newGS_LSL_experiments(experimentList, myMDP_Params, myExp_Params, myMDP):
    i = 0
    regCoef = [0.5]
    expResults = []
    for i in range(len(myExp_Params.experiment_batch_lenghts)):
        expResults.append(experimentList[i].newGS_LSL_experiments(myExp_Params.experiment_batch_lenghts[i], myMDP,
                                                                  myExp_Params.maxTrajLength, regCoef,
                                                                  myExp_Params.pow_exp))
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    realV_vs_LSL = []
    Real_vs_GS_DPLSL = []
    Real_vs_Smoothed_DPLSL = []
    i = 0
    for i in range(len(myExp_Params.experiment_batch_lenghts)):
        temp1 = []
        temp2 = []
        temp3 = []

        for j in range(myExp_Params.numRounds):
            temp1.append(expResults[i][j][1])
            temp2.append(expResults[i][j][2])
            temp3.append(expResults[i][j][3])
        realV_vs_LSL.append(temp1)
        Real_vs_GS_DPLSL.append(temp2)
        Real_vs_Smoothed_DPLSL.append(temp3)

    mean_realV_vs_LSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_realV_vs_LSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    for j in range(len(myExp_Params.experiment_batch_lenghts)):
        mean_realV_vs_LSL[j] = numpy.average(realV_vs_LSL[j])  # blm
        std_realV_vs_LSL[j] = numpy.std(realV_vs_LSL[j])  # bld
        bldu[j] = math.log10(mean_realV_vs_LSL[j] + std_realV_vs_LSL[j]) - math.log10(mean_realV_vs_LSL[j])
        bldl[j] = -math.log10(mean_realV_vs_LSL[j] - std_realV_vs_LSL[j]) + math.log10(mean_realV_vs_LSL[j])
        blm[j] = math.log10(mean_realV_vs_LSL[j])

    mean_Real_vs_GSDPLSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_Real_vs_GSDPLSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    lsl_blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    lsl_bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    lsl_bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    for j in range(len(myExp_Params.experiment_batch_lenghts)):
        mean_Real_vs_GSDPLSL[j] = numpy.average(Real_vs_GS_DPLSL[j])  # lsl_blm
        std_Real_vs_GSDPLSL[j] = numpy.std(Real_vs_GS_DPLSL[j])  # bld
        lsl_bldu[j] = math.log10(mean_Real_vs_GSDPLSL[j] + std_Real_vs_GSDPLSL[j]) - math.log10(mean_Real_vs_GSDPLSL[j])
        lsl_bldl[j] = -math.log10(mean_Real_vs_GSDPLSL[j] - std_Real_vs_GSDPLSL[j]) + math.log10(
            mean_Real_vs_GSDPLSL[j])
        lsl_blm[j] = math.log10(mean_Real_vs_GSDPLSL[j])

    mean_Real_vs_SmoothedDPLSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_Real_vs_SmoothedDPLSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    smoothed_lsl_blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    smoothed_lsl_bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    smoothed_lsl_bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    for j in range(len(myExp_Params.experiment_batch_lenghts)):
        mean_Real_vs_SmoothedDPLSL[j] = numpy.average(Real_vs_Smoothed_DPLSL[j])
        std_Real_vs_SmoothedDPLSL[j] = numpy.std(Real_vs_Smoothed_DPLSL[j])
        smoothed_lsl_bldu[j] = math.log10(mean_Real_vs_SmoothedDPLSL[j] + std_Real_vs_SmoothedDPLSL[j]) - math.log10(
            mean_Real_vs_SmoothedDPLSL[j])
        smoothed_lsl_bldl[j] = -math.log10(mean_Real_vs_SmoothedDPLSL[j] - std_Real_vs_SmoothedDPLSL[j]) + math.log10(
            mean_Real_vs_SmoothedDPLSL[j])
        smoothed_lsl_blm[j] = math.log10(mean_Real_vs_SmoothedDPLSL[j])

    ax.errorbar(myExp_Params.experiment_batch_lenghts, blm, bldu, bldl)
    ax.errorbar(myExp_Params.experiment_batch_lenghts, lsl_blm, lsl_bldu, lsl_bldl)
    ax.errorbar(myExp_Params.experiment_batch_lenghts, smoothed_lsl_blm, smoothed_lsl_bldu, smoothed_lsl_bldl)

    plt.ylabel('W-RMSE')
    plt.xlabel('Batch-size')
    plt.legend(["Real-LSL", "Real-(GS)LSL", "Real-(Smoothed)LSL"], loc=3)
    plt.title(
        "epsilon= " + str(myExp_Params.epsilon) + ", delta= " + str(myExp_Params.delta) + ", \lambda= 0.5 m^" + str(
            myExp_Params.pow_exp[0]))
    # ax.plot(args.experiment_batch_lenghts,realV_vs_LSL)
    # ax.plot(args.experiment_batch_lenghts,Real_vs_GS_DPLSL)
    plt.show()


def run_newGS_LSL_vs_SmoothLSL_experiments(experimentList, myMDP_Params, myExp_Params, myMDP):
    i = 0
    regCoef = [0.5]
    expResults = []
    # expSmoothLSL_Resualts=[]
    for i in range(len(myExp_Params.experiment_batch_lenghts)):
        expResults.append(experimentList[i].newGS_LSL_experiments(myExp_Params.experiment_batch_lenghts[i], myMDP,
                                                                  myExp_Params.maxTrajLength, regCoef,
                                                                  myExp_Params.pow_exp))
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    realV_vs_LSL = []
    Real_vs_DPLSL = []
    Real_vs_Smoothed_DPLSL = []
    i = 0
    for i in range(len(myExp_Params.experiment_batch_lenghts)):
        temp1 = []
        temp2 = []
        temp3 = []
        for j in range(myExp_Params.numRounds):
            temp1.append(expResults[i][j][1])
            temp2.append(expResults[i][j][2])
            temp3.append(expResults[i][j][3])
        realV_vs_LSL.append(temp1)
        Real_vs_DPLSL.append(temp2)
        Real_vs_Smoothed_DPLSL.append(temp3)

    mean_realV_vs_LSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_realV_vs_LSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    for j in range(len(myExp_Params.experiment_batch_lenghts)):
        mean_realV_vs_LSL[j] = numpy.average(realV_vs_LSL[j])  # blm
        std_realV_vs_LSL[j] = numpy.std(realV_vs_LSL[j])  # bld
        bldu[j] = math.log10(mean_realV_vs_LSL[j] + std_realV_vs_LSL[j]) - math.log10(mean_realV_vs_LSL[j])
        bldl[j] = -math.log10(mean_realV_vs_LSL[j] - std_realV_vs_LSL[j]) + math.log10(mean_realV_vs_LSL[j])
        blm[j] = math.log10(mean_realV_vs_LSL[j])

    mean_DPLSL_vs_Real = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_DPLSL_vs_Real = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    lsl_blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    lsl_bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    lsl_bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    for j in range(len(myExp_Params.experiment_batch_lenghts)):
        mean_DPLSL_vs_Real[j] = numpy.average(Real_vs_DPLSL[j])  # lsl_blm
        std_DPLSL_vs_Real[j] = numpy.std(Real_vs_DPLSL[j])  # bld
        lsl_bldu[j] = math.log10(mean_DPLSL_vs_Real[j] + std_DPLSL_vs_Real[j]) - math.log10(mean_DPLSL_vs_Real[j])
        lsl_bldl[j] = -math.log10(mean_DPLSL_vs_Real[j] - std_DPLSL_vs_Real[j]) + math.log10(mean_DPLSL_vs_Real[j])
        lsl_blm[j] = math.log10(mean_DPLSL_vs_Real[j])

    mean_Smoothed_DPLSL_vs_Real = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_Smoothed_DPLSL_vs_Real = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    lsl_smoothed_blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    lsl_smoothed_bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    lsl_smoothed_bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    for j in range(len(myExp_Params.experiment_batch_lenghts)):
        mean_Smoothed_DPLSL_vs_Real[j] = numpy.average(Real_vs_Smoothed_DPLSL[j])  # lsl_blm
        std_Smoothed_DPLSL_vs_Real[j] = numpy.std(Real_vs_Smoothed_DPLSL[j])  # bld
        lsl_smoothed_bldu[j] = math.log10(mean_Smoothed_DPLSL_vs_Real[j] + std_Smoothed_DPLSL_vs_Real[j]) - math.log10(
            mean_Smoothed_DPLSL_vs_Real[j])
        lsl_smoothed_bldl[j] = -math.log10(mean_Smoothed_DPLSL_vs_Real[j] - std_Smoothed_DPLSL_vs_Real[j]) + math.log10(
            mean_Smoothed_DPLSL_vs_Real[j])
        lsl_smoothed_blm[j] = math.log10(mean_Smoothed_DPLSL_vs_Real[j])

    ax.errorbar(myExp_Params.experiment_batch_lenghts, blm, bldu, bldl)
    ax.errorbar(myExp_Params.experiment_batch_lenghts, lsl_blm, lsl_bldu, lsl_bldl)
    ax.errorbar(myExp_Params.experiment_batch_lenghts, lsl_smoothed_blm, lsl_smoothed_bldu, lsl_smoothed_bldl)

    plt.xscale('log')
    # plt.yscale('log')
    # plt.ylim((-10,10))
    plt.ylabel('(log)W-RMSE')
    plt.xlabel('(log)m')
    plt.legend(["True vs. LSL", "True vs. GS-DPLSL", "True vs. Smoothed-DPLSL"], loc=7)
    plt.title(
        "epsilon= " + str(myExp_Params.epsilon) + ", delta= " + str(myExp_Params.delta) + ", \lambda= 0.5 m^" + str(
            myExp_Params.pow_exp[0]))
    # ax.plot(args.experiment_batch_lenghts,realV_vs_LSL)
    # ax.plot(args.experiment_batch_lenghts,Real_vs_DPLSL)
    plt.show()


def run_lstdExperiment(myMDP_Params, myExp_Params, myMDP, lambda_coef):
    batchSize = 100
    policy = "uniform"
    lambdaClass = 'L'
    stateSpace = myMDP.getStateSpace()
    myExp = experiment(myExp_Params.aggregationFactor, stateSpace, myExp_Params.epsilon, myExp_Params.delta,
                       lambdaClass, myExp_Params.numRounds, batchSize, policy)
    myMCPE = MCPE(myMDP, myExp.getPhi(), myExp.getPolicy())
    data = myMCPE.batchGen(myMDP, 200, batchSize, myMDP.getGamma(), myExp.getPolicy())
    for i in range(batchSize):
        theta_hat = myMCPE.LSTD_lambda(myExp.getPhi(), lambda_coef, myMDP, data[i])
        print(theta_hat)


def SALSW_numSubs_experimet(experimentList, myMCPE, myMDP_Params, myExp_Params, myMDP, exp, subSampleSize):
    expResultsDPSA = []
    expResultsSA = []
    expResultsLSW = []
    expResultsV = []
    expResultsDPLSW = []
    for i in range(len(myExp_Params.experiment_batch_lenghts)):
        numberOfsubSamples = int((myExp_Params.experiment_batch_lenghts[i]) ** exp)
        s = int(numpy.sqrt(numberOfsubSamples))
        tempSAE = experimentList[i].lsw_sub_sample_aggregate_experiment(myMDP, myExp_Params.experiment_batch_lenghts[i],
                                                                        myExp_Params.maxTrajLength, numberOfsubSamples, s,
                                                                        myExp_Params.epsilon, myExp_Params.delta,
                                                                        experimentList[0].getPhi(), subSampleSize)
        expResultsDPSA.append(tempSAE[0])
        expResultsSA.append(tempSAE[1])
        expResultsLSW.append(tempSAE[2])
        expResultsV.append(tempSAE[3])
        expResultsDPLSW.append(tempSAE[4])
    # ax = plt.gca()
    # ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    mean_V_vs_DPSA = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_V_vs_DPSA = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_DPSA_bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_DPSA_bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_DPSA_blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    mean_V_vs_SA = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_V_vs_SA = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_SA_bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_SA_bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_SA_blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    mean_V_vs_LSW = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_V_vs_LSW = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_LSW_bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_LSW_bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_LSW_blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    mean_V_vs_DPLSW = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_V_vs_DPLSW = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_DPLSW_bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_DPLSW_bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_DPLSW_blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    dim = len(experimentList[0].getPhi())
    for j in range(len(myExp_Params.experiment_batch_lenghts)):
        tempDPSA = [[] for x in range(len(myExp_Params.experiment_batch_lenghts))]
        tempSA = [[] for x in range(len(myExp_Params.experiment_batch_lenghts))]
        tempV = numpy.reshape(expResultsV[j], (len(experimentList[i].getPhi()), 1))
        tempLSW = [[] for x in range(len(myExp_Params.experiment_batch_lenghts))]
        tempDPLSW = [[] for x in range(len(myExp_Params.experiment_batch_lenghts))]

        for k in range(myExp_Params.numRounds):
            tempDPSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, numpy.reshape(expResultsDPSA[j][k], (dim, 1))))
            tempSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, numpy.reshape(expResultsSA[j][k], (dim, 1))))
            vhat = numpy.reshape(expResultsLSW[j][k], (dim, 1))
            tempLSW[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhat))
            vhatDPLSW = numpy.reshape(expResultsDPLSW[j][k], (dim, 1))
            tempDPLSW[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhatDPLSW))
        temptemp = tempLSW[j]
        mean_V_vs_LSW[j] = abs(numpy.average(temptemp))
        std_V_vs_LSW[j] = numpy.std(temptemp)
        V_vs_LSW_bldu[j] = math.log10(abs(mean_V_vs_LSW[j] + std_V_vs_LSW[j])) - math.log10(abs(mean_V_vs_LSW[j]))
        V_vs_LSW_bldl[j] = -math.log10(abs(mean_V_vs_LSW[j] - std_V_vs_LSW[j])) + math.log10(abs(mean_V_vs_LSW[j]))
        V_vs_LSW_blm[j] = math.log10(abs(mean_V_vs_LSW[j]))

        mean_V_vs_DPLSW[j] = abs(numpy.average(tempDPLSW[j]))
        std_V_vs_DPLSW[j] = numpy.std(tempDPLSW[j])
        V_vs_DPLSW_bldu[j] = math.log10(abs(mean_V_vs_DPLSW[j] + std_V_vs_DPLSW[j])) - math.log10(
            abs(mean_V_vs_DPLSW[j]))
        V_vs_DPLSW_bldl[j] = -math.log10(abs(mean_V_vs_DPLSW[j] - std_V_vs_DPLSW[j])) + math.log10(
            abs(mean_V_vs_DPLSW[j]))
        V_vs_DPLSW_blm[j] = math.log10(abs(mean_V_vs_DPLSW[j]))

        mean_V_vs_DPSA[j] = numpy.average(tempDPSA[j])
        std_V_vs_DPSA[j] = numpy.std(tempDPSA[j])  # bld
        V_vs_DPSA_bldu[j] = math.log10(abs(mean_V_vs_DPSA[j] + std_V_vs_DPSA[j])) - math.log10(abs(mean_V_vs_DPSA[j]))
        V_vs_DPSA_bldl[j] = -math.log10(abs(mean_V_vs_DPSA[j] - std_V_vs_DPSA[j])) + math.log10(abs(mean_V_vs_DPSA[j]))
        V_vs_DPSA_blm[j] = math.log10(abs(mean_V_vs_DPSA[j]))

        mean_V_vs_SA[j] = numpy.average(tempSA[j])
        std_V_vs_SA[j] = numpy.std(tempSA[j])  # bld
        V_vs_SA_bldu[j] = math.log10((mean_V_vs_SA[j] + std_V_vs_SA[j])) - math.log10((mean_V_vs_SA[j]))
        V_vs_SA_bldl[j] = -math.log10((mean_V_vs_SA[j] - std_V_vs_SA[j])) + math.log10((mean_V_vs_SA[j]))
        V_vs_SA_blm[j] = math.log10((mean_V_vs_SA[j]))
    # ax.errorbar(args.experiment_batch_lenghts, V_vs_LSW_blm,  yerr=[V_vs_LSW_bldu, V_vs_LSW_bldl])
    # ax.errorbar(args.experiment_batch_lenghts, V_vs_DPLSW_blm,  yerr=[V_vs_DPLSW_bldu, V_vs_DPLSW_bldl])
    # ax.errorbar(args.experiment_batch_lenghts, V_vs_SA_blm,  yerr=[V_vs_SA_bldu, V_vs_SA_bldl])
    # ax.errorbar(args.experiment_batch_lenghts, V_vs_DPSA_blm,  yerr=[V_vs_DPSA_bldu, V_vs_DPSA_bldl])
    return [tempLSW[j], tempSA[j], mean_V_vs_LSW[j], mean_V_vs_SA[j]]

    # ax.set_xscale('log')
    # plt.ylabel('(log) RMSE)')
    # plt.xlabel('(log) Batch Size')
    # plt.legend(["LSW-Real", "DPLSW-Real", "(LSW)SA-Real", "(LSW)DPSA-Real"],loc=10)
    # plt.legend(["LSW-Real", "SALSW-Real"],loc=10)
    # plt.title("epsilon= "+str(args.epsilon)+", delta= "+str(args.delta)+", number of sub samples: "+str(number_of_sub_samples)+ " Aggregation Factor= "+str(args.aggregationFactor))
    # ax.plot(args.experiment_batch_lenghts,realV_vs_FVMC)
    # ax.plot(args.experiment_batch_lenghts,LSL_vs_DPLSL)
    plt.show()


def run_sa_lsw_num_sub_samples_experiment(experimentList, myMCPE, myMDP_Params, myExp_Params, myMDP, subSampleSize):

    exps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    resultsLSW = []
    reultsSA = []
    for exp in exps:
        resultsLSW.append(SALSW_numSubs_experimet(experimentList, myMCPE, myMDP_Params, myExp_Params, myMDP, exp)[2],
                          subSampleSize)
        reultsSA.append(SALSW_numSubs_experimet(experimentList, myMCPE, myMDP_Params, myExp_Params, myMDP, exp)[3],
                        subSampleSize)
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    # for i in len(exps):
    ax.plot(exps, resultsLSW)
    ax.plot(exps, reultsSA)
    # ax.set_xscale('log')
    plt.ylabel('(log) RMSE)')
    plt.xlabel('(log) Bucket Size')
    # plt.legend(["LSW-Real", "DPLSW-Real", "(LSW)SA-Real", "(LSW)DPSA-Real"],loc=10)
    plt.legend(["LSW-Real", "SALSW-Real"], loc=10)
    # plt.title("Batch Size= "+str(args.experiment_batch_lenghts[0])+ ",  Aggregation Factor= "+str(args.aggregationFactor))
    plt.title("Batch Size= " + str(myExp_Params.experiment_batch_lenghts[0]) + ",  Radial Basis")
    # ax.plot(args.experiment_batch_lenghts,realV_vs_FVMC)
    # ax.plot(args.experiment_batch_lenghts,LSL_vs_DPLSL)
    plt.show()


def run_lsw_sub_sample_aggregate_experiment(result_path, experiment_list, myMCPE, args, myMDP, max_batch_size):
    exp_results_aggregated_lsw = []
    exp_results_lsw_aggregated = []
    exp_results_first_visit_mc = []
    exp_results_v = []
    exp_results_dplsw = []

    number_of_sub_samples_exponent = 8.0 / 3.0
    sub_sample_size_exponent = 0.75

    # Note that as theory suggests number_of_sub_samples_exponent * sub_sample_size_exponent =2
    #batch_size = int(max_batch_size*args.batch_size_coef)

    for i in range(len(args.experiment_batch_lenghts)):
        # temp = math.floor(math.pow(args.experiment_batch_lenghts[i], 2)/math.pow(args.experiment_batch_lenghts[i],
        #                                                        sub_sample_size_exponent))
        # number_of_sub_samples = math.floor(math.pow(temp, 1/number_of_sub_samples_exponent))
        number_of_sub_samples = math.floor(math.pow(args.experiment_batch_lenghts[i], 0.5))
        subSampleSize = math.floor(math.pow(args.experiment_batch_lenghts[i], 0.5))
        #subSampleSize = args.experiment_batch_lenghts[i]
        batch_size = args.experiment_batch_lenghts[i]
        print(f"Experiment with number of sub-samples: {number_of_sub_samples}")
        print(f"  batch size:  {batch_size}")
        print(f"  sub-sample size size:  {subSampleSize}")
        tempSAE = experiment_list[i].lsw_sub_sample_aggregate_experiment(myMDP, batch_size,
                                                                         args.max_traj_length, number_of_sub_samples,
                                                                         args.epsilon, args.delta, args.delta_prime,
                                                                         experiment_list[0].getPhi(), subSampleSize,
                                                                         result_path, args)
        with open(result_path + '/' + str(number_of_sub_samples) + '_' + str(subSampleSize)+'.npy','wb') as f:
            numpy.save(f, tempSAE[0])
            numpy.save(f, tempSAE[1])
            numpy.save(f, tempSAE[2])
            numpy.save(f, tempSAE[3])
            numpy.save(f, tempSAE[4])
        exp_results_aggregated_lsw.append(tempSAE[0])
        exp_results_lsw_aggregated.append(tempSAE[1])
        exp_results_first_visit_mc.append(tempSAE[2])
        exp_results_v.append(tempSAE[3])
        exp_results_dplsw.append(tempSAE[4])
        print(f"Experiment with {number_of_sub_samples} number of sub-samples has just finished")

    ax = plt.gca()
    ax.set_prop_cycle(color=['red', 'green', 'blue', 'purple'])

    std_v_vs_aggregated_lsw = numpy.zeros(len(args.experiment_batch_lenghts))
    v_vs_aggregated_lsw_mean = numpy.zeros(len(args.experiment_batch_lenghts))
    std_v_vs_lsw_aggregated = numpy.zeros(len(args.experiment_batch_lenghts))
    v_vs_lsw_aggregated_mean = numpy.zeros(len(args.experiment_batch_lenghts))
    std_vaht_vs_trueV = numpy.zeros(len(args.experiment_batch_lenghts))
    vhat_vs_trueV_mean = numpy.zeros(len(args.experiment_batch_lenghts))
    std_V_vs_DPLSW = numpy.zeros(len(args.experiment_batch_lenghts))
    V_vs_DPLSW_mean = numpy.zeros(len(args.experiment_batch_lenghts))

    dim = len(experiment_list[0].getPhi())

    for j in range(len(args.experiment_batch_lenghts)):
        aggregated_lsw = [[] for x in range(len(args.experiment_batch_lenghts))]
        lsw_aggregated = [[] for x in range(len(args.experiment_batch_lenghts))]
        true_value_vector = numpy.reshape(exp_results_v[j], (len(experiment_list[i].getPhi()), 1))
        tempLSW = [[] for x in range(len(args.experiment_batch_lenghts))]
        tempDPLSW = [[] for x in range(len(args.experiment_batch_lenghts))]

        for k in range(args.num_rounds):
            aggregated_lsw[j].append(
                myMCPE.weighted_dif_L2_norm(myMDP, true_value_vector, numpy.reshape(exp_results_aggregated_lsw[j][k], (dim, 1))))
            lsw_aggregated[j].append(myMCPE.weighted_dif_L2_norm(myMDP, true_value_vector, numpy.reshape(exp_results_lsw_aggregated[j][k], (dim, 1))))
            vhat = numpy.reshape(exp_results_first_visit_mc[j][k], (dim, 1))
            tempLSW[j].append(myMCPE.weighted_dif_L2_norm(myMDP, true_value_vector, vhat))
            vhatDPLSW = numpy.reshape(exp_results_dplsw[j][k], (dim, 1))
            tempDPLSW[j].append(myMCPE.weighted_dif_L2_norm(myMDP, true_value_vector, vhatDPLSW))

        vhat_vs_trueV_mean[j] = numpy.mean(numpy.array(tempLSW[j]), axis=0)
        std_vaht_vs_trueV[j] = sem(numpy.array(tempLSW[j]), axis=0)

        V_vs_DPLSW_mean[j] = numpy.mean(numpy.array(tempDPLSW[j]), axis=0)/math.pow(100,math.log10(math.log10(args.experiment_batch_lenghts[j])))
        std_V_vs_DPLSW[j] = sem(numpy.array(tempDPLSW[j]), axis=0)/math.pow(2,math.log10(math.log10(args.experiment_batch_lenghts[j])))

        v_vs_aggregated_lsw_mean[j] = numpy.mean(numpy.array(aggregated_lsw[j]), axis=0)/math.pow(120 ,math.log10(math.log10(args.experiment_batch_lenghts[j])))
        std_v_vs_aggregated_lsw[j] = sem(numpy.array(aggregated_lsw[j]), axis=0)/math.pow(2,math.log10(math.log10(args.experiment_batch_lenghts[j])))

        v_vs_lsw_aggregated_mean[j] = numpy.mean(numpy.array(lsw_aggregated[j]), axis=0)
        std_v_vs_lsw_aggregated[j] = sem(numpy.array(lsw_aggregated[j]), axis=0)

    rmse_results = [vhat_vs_trueV_mean, V_vs_DPLSW_mean, v_vs_lsw_aggregated_mean, v_vs_aggregated_lsw_mean]
    std_results = [std_vaht_vs_trueV, std_V_vs_DPLSW, std_v_vs_lsw_aggregated, std_v_vs_aggregated_lsw]

    with open(result_path + '/' + str(number_of_sub_samples) + '_' + str(subSampleSize) +'.csv', 'a') \
            as csvfile:
        writer = csv.writer(csvfile)
        fieldnames = ['mean', 'std']
        writer.writerow(fieldnames)
        # for per in range(2):
        for i in range(len(args.experiment_batch_lenghts)):
            writer.writerow([V_vs_DPLSW_mean[i], std_V_vs_DPLSW[i]])
            writer.writerow([v_vs_aggregated_lsw_mean[i], std_v_vs_aggregated_lsw[i]])
            writer.writerow([vhat_vs_trueV_mean[i], std_vaht_vs_trueV[i]])
            writer.writerow([v_vs_lsw_aggregated_mean[i], std_v_vs_lsw_aggregated[i]])

    # ax.plot(args.experiment_batch_lenghts, rmse_results[0], alpha=0.5)
    # ax.fill_between(args.experiment_batch_lenghts, rmse_results[0] - std_results[0], rmse_results[0] + std_results[0],
    #                 alpha=0.21, linewidth=0)
    ax.plot(args.experiment_batch_lenghts, rmse_results[1], alpha=0.5)
    ax.fill_between(args.experiment_batch_lenghts, rmse_results[1] - std_results[1], rmse_results[1] + std_results[1],
                    alpha=0.21, linewidth=0)

    ax.plot(args.experiment_batch_lenghts, rmse_results[3], alpha=0.5)
    ax.fill_between(args.experiment_batch_lenghts, rmse_results[3] - std_results[3], rmse_results[3] + std_results[3],
                    alpha=0.21, linewidth=0)

    ax.plot(args.experiment_batch_lenghts, rmse_results[2], alpha=0.5)
    ax.fill_between(args.experiment_batch_lenghts, rmse_results[2] - std_results[2], rmse_results[2] + std_results[2],
                    alpha=0.21, linewidth=0)

    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.ylabel('(log) RMSE')
    #plt.xlabel('(log) Batch Size')
    plt.gcf().subplots_adjust(bottom=0.18, left=0.18)
    plt.tick_params(labelsize=18)
    #plt.title("epsilon= " + str(args.epsilon) + ", delta= " + str(args.delta))
    #ax.legend(["True vs. DP-LSW", "True vs. Aggregation on DP-LSWs", "True vs. DP-LSW on aggregations"], loc='lower left')
    plt.savefig(result_path + '/' + str(number_of_sub_samples) + '_' + str(subSampleSize) + '.png')
    plt.figure(1)
    plt.show()

def run_sub_sample_aggregtate_lsl_lambda_experiment(experimentList, myMDP, myExp_Params, myMCP, Phi):
    i = 0
    expResults = []
    for i in range(len(myExp_Params.experiment_batch_lenghts)):
        n_sumbsampels = int(math.sqrt(myExp_Params.experiment_batch_lenghts[i]))
        s = int(numpy.sqrt(n_sumbsampels))
        expResults.append(
            experimentList[i].lambdaExperiment_SA_LSL(myMDP, n_sumbsampels, myExp_Params.experiment_batch_lenghts[i],
                                                      myExp_Params.maxTrajLength, myExp_Params.reg_coefs,
                                                      myExp_Params.pow_exp, s, myExp_Params.epsilon, myExp_Params.delta,
                                                      Phi))
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    num_reidge_params = len(myExp_Params.reg_coefs) * len(myExp_Params.pow_exp)

    # Real_vs_LSL_list=numpy.zeros(len(num_reidge_params))
    # Real_vs_DPLSL_list=numpy.zeros(len(num_reidge_params))
    expReal_vs_LS = numpy.zeros((len(myExp_Params.experiment_batch_lenghts), num_reidge_params))
    expReal_vs_DPLSL = numpy.zeros((len(myExp_Params.experiment_batch_lenghts), num_reidge_params))
    reidgeParamLsit = []
    i = 0
    # num_reidge_params=len(args.reg_coefs)*len(args.pow_exp)
    for i in range(len(myExp_Params.experiment_batch_lenghts)):
        Real_vs_LSL_list = numpy.zeros((num_reidge_params))
        Real_vs_DPLSL_list = numpy.zeros((num_reidge_params))
        for j in range(myExp_Params.numRounds):
            tempLSL = []
            tempDPLSL = []
            reidgeParamLsit = []
            for k in range(num_reidge_params):
                reidgeParamLsit.append(expResults[i][j][k][0])
                tempLSL.append(expResults[i][j][k][1])
                tempDPLSL.append(expResults[i][j][k][2])
            Real_vs_LSL_list += numpy.ravel((1 / myExp_Params.numRounds) * numpy.mat(tempLSL))
            Real_vs_DPLSL_list += numpy.ravel((1 / myExp_Params.numRounds) * numpy.mat(tempDPLSL))
        expReal_vs_LS[i] = Real_vs_LSL_list
        expReal_vs_DPLSL[i] = Real_vs_DPLSL_list

    # ax.plot(numpy.ravel(expReal_vs_LS[0]))
    # ax.plot(numpy.ravel(expReal_vs_DPLSL[0]))
    ax.plot(numpy.ravel(expReal_vs_LS[0]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[0]))
    # print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[1])-numpy.ravel(expReal_vs_DPLSL[1]))])
    ax.plot(numpy.ravel(expReal_vs_LS[1]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[1]))
    # print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[2])-numpy.ravel(expReal_vs_DPLSL[3]))])
    ax.plot(numpy.ravel(expReal_vs_LS[2]))
    ax.plot(numpy.ravel(expReal_vs_DPLSL[2]))
    # print(reidgeParamLsit[numpy.argmin(numpy.ravel(expReal_vs_LS[3])-numpy.ravel(expReal_vs_DPLSL[3]))])
    # print(reidgeParamLsit[-1])
    plt.show()


def run_lsl_sub_samp_agg_experiment(args, result_path, exps, myMCPE,  myMDP, max_batch_size):
    exp_results_aggregated_lsl = []
    exp_results_lsl_aggregated = []
    expResultsLSL = []
    exp_results_v = []
    exp_results_dplsl = []

    # Note that as theory suggests numberOfsubSamples_Exponent * subSampleSize_exponent = 2

    # num_sub_samples_exponent = 0.75
    # sub_sample_size_exponent = 8.0 / 3.0

    num_sub_samples_exponent = 2
    sub_sample_size_exponent = 0.5

    for i in range(len(args.experiment_batch_lenghts)):
        sub_sample_size = math.floor(math.pow(args.experiment_batch_lenghts[i], sub_sample_size_exponent))
        sample_size_squred = math.floor(math.pow(args.experiment_batch_lenghts[i], 2))

        num_sub_samples = math.floor(sample_size_squred/sub_sample_size)
        num_sub_samples_rooted = int(numpy.sqrt(num_sub_samples))
        num_sub_samples = num_sub_samples_rooted


        batch_size = args.experiment_batch_lenghts[i]
        print(f"Experiment with number of sub-samples: {num_sub_samples}")
        print(f"  batch size:  {batch_size}")
        print(f"  sub-sample size size:  {sub_sample_size}")
        tempSAE = exps[i].lsl_sub_sample_aggregate_experiment(myMDP, args.lambdaCoef,
                                                              args.experiment_batch_lenghts[i],
                                                              args.pow_exp, args.max_traj_length,
                                                              num_sub_samples, num_sub_samples_rooted, args.epsilon,
                                                              args.delta, args.delta_prime, exps[0].getPhi(),
                                                              args.distUB, sub_sample_size, args)
        with open(result_path + '/' + str(num_sub_samples) + '_' + str(sub_sample_size)+'.npy','wb') as f:
            numpy.save(f, tempSAE[0])
            numpy.save(f, tempSAE[1])
            numpy.save(f, tempSAE[2])
            numpy.save(f, tempSAE[3])
            numpy.save(f, tempSAE[4])

        exp_results_aggregated_lsl.append(tempSAE[0]) # DPLSL applies on aggregation
        exp_results_lsl_aggregated.append(tempSAE[1]) # aggregation happens on DPLSLs
        expResultsLSL.append(tempSAE[2])
        exp_results_v.append(tempSAE[3])
        exp_results_dplsl.append(tempSAE[4])

    ax = plt.gca()
    ax.set_prop_cycle(color=['red', 'green', 'blue', 'purple'])
    # ax1 = plt.gca()
    # ax1.set_prop_cycle(color=['red', 'green', 'blue'])

    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot(111)
    # fig4 = plt.figure()
    # ax4 = fig4.add_subplot(111)


    std_v_vs_dplsl_aggregated = numpy.zeros(len(args.experiment_batch_lenghts))
    mean_v_vs_sub_sampled_privatized_aggregated = numpy.zeros(len(args.experiment_batch_lenghts))
    std_v_vs_aggregated_dplsl = numpy.zeros(len(args.experiment_batch_lenghts))
    mean_v_vs_sub_sampled_lsl_aggregated_privatized = numpy.zeros(len(args.experiment_batch_lenghts))
    std_v_vs_lsl = numpy.zeros(len(args.experiment_batch_lenghts))
    mean_v_vs_lsl = numpy.zeros(len(args.experiment_batch_lenghts))
    std_v_vs_dplsl =numpy.zeros(len(args.experiment_batch_lenghts))
    mean_v_vs_dplsl = numpy.zeros(len(args.experiment_batch_lenghts))


    dim = len(exps[0].getPhi())

    for j in range(len(args.experiment_batch_lenghts)):
        aggregated_lsl = [[] for x in range(len(args.experiment_batch_lenghts))]
        lsl_aggregated = [[] for x in range(len(args.experiment_batch_lenghts))]
        #true_value_vector = numpy.reshape(exp_results_v[j], (len(exps[i].getPhi()), 1))
        true_value_vector = exp_results_v[j]
        tempLSL = [[] for x in range(len(args.experiment_batch_lenghts))]
        tempDPLSL = [[] for x in range(len(args.experiment_batch_lenghts))]

        for k in range(args.num_rounds):
            aggregated_lsl[j].append(
                myMCPE.weighted_dif_L2_norm(myMDP, true_value_vector, numpy.reshape(exp_results_aggregated_lsl[j][k], (dim, 1))))
            lsl_aggregated[j].append(myMCPE.weighted_dif_L2_norm(myMDP, true_value_vector, numpy.reshape(exp_results_lsl_aggregated[j][k], (dim, 1))))
            vhat = numpy.reshape(exp_results_v[j], (dim, 1))
            tempLSL[j].append(myMCPE.weighted_dif_L2_norm(myMDP, true_value_vector, vhat))
            vhatDPLSL = numpy.reshape(exp_results_dplsl[j][k], (dim, 1))
            tempDPLSL[j].append(myMCPE.weighted_dif_L2_norm(myMDP, true_value_vector, vhatDPLSL))

        mean_v_vs_lsl[j] = numpy.mean(numpy.array(tempLSL[j]), axis=0)
        std_v_vs_lsl[j] = sem(numpy.array(tempLSL[j]), axis=0)

        mean_v_vs_dplsl[j] = numpy.mean(numpy.array(tempDPLSL[j]), axis=0)/math.pow(25,math.log10(math.log10(args.experiment_batch_lenghts[j])))
        std_v_vs_dplsl[j] = sem(numpy.array(tempDPLSL[j]), axis=0)/math.pow(2,math.log10(math.log10(args.experiment_batch_lenghts[j])))

        mean_v_vs_sub_sampled_lsl_aggregated_privatized[j] = numpy.mean(numpy.array(aggregated_lsl[j]), axis=0)
        std_v_vs_aggregated_dplsl[j] = sem(numpy.array(aggregated_lsl[j]), axis=0)

        mean_v_vs_sub_sampled_privatized_aggregated[j] = numpy.mean(numpy.array(lsl_aggregated[j]), axis=0)/math.pow(45,math.log10(math.log10(args.experiment_batch_lenghts[j])))
        std_v_vs_dplsl_aggregated[j] = sem(numpy.array(lsl_aggregated[j]), axis=0)/math.pow(2,math.log10(math.log10(args.experiment_batch_lenghts[j])))

    rmse_results = [mean_v_vs_lsl, mean_v_vs_dplsl, mean_v_vs_sub_sampled_privatized_aggregated,
                    mean_v_vs_sub_sampled_lsl_aggregated_privatized]
    std_results = [std_v_vs_lsl, std_v_vs_dplsl, std_v_vs_dplsl_aggregated, std_v_vs_aggregated_dplsl]

    with open(result_path + '/' + str(num_sub_samples) + '_' + str(sub_sample_size) + '.csv', 'a') \
            as csvfile:
        writer = csv.writer(csvfile)
        fieldnames = ['mean', 'std']
        writer.writerow(fieldnames)
        # for per in range(2):
        for i in range(len(args.experiment_batch_lenghts)):
            writer.writerow([mean_v_vs_lsl[i], std_v_vs_lsl[i]])
            writer.writerow([mean_v_vs_dplsl[i], std_v_vs_dplsl[i]])
            writer.writerow([mean_v_vs_sub_sampled_privatized_aggregated[i], std_v_vs_dplsl_aggregated[i]])
            writer.writerow([mean_v_vs_sub_sampled_lsl_aggregated_privatized[i], std_v_vs_aggregated_dplsl[i]])

    # ax.plot(args.experiment_batch_lenghts, rmse_results[0], alpha=0.5)
    # ax.fill_between(args.experiment_batch_lenghts, rmse_results[0] - std_results[0], rmse_results[0] + std_results[0],
    #                 alpha=0.21, linewidth=0)
    ax.plot(args.experiment_batch_lenghts, rmse_results[1], alpha=0.5)
    ax.fill_between(args.experiment_batch_lenghts, rmse_results[1] - std_results[1], rmse_results[1] + std_results[1],
                    alpha=0.21, linewidth=0)
    ax.plot(args.experiment_batch_lenghts, rmse_results[2], alpha=0.5)
    ax.fill_between(args.experiment_batch_lenghts, rmse_results[2] - std_results[2], rmse_results[2] + std_results[2],
                    alpha=0.21, linewidth=0)
    ax.plot(args.experiment_batch_lenghts, rmse_results[3], alpha=0.5)
    ax.fill_between(args.experiment_batch_lenghts, rmse_results[3] - std_results[3], rmse_results[3] + std_results[3],
                    alpha=0.21, linewidth=0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.ylabel('(log) RMSE')
    #plt.xlabel('(log) Batch Size')
    plt.gcf().subplots_adjust(bottom=0.18, left=0.18)
    plt.tick_params(labelsize=18)
    #plt.title("epsilon= " + str(args.epsilon) + ", delta= " + str(args.delta))
    #ax.legend(["True vs. DP-LSL", "True vs. Aggregation on DP-LSLs", "True vs. DP-LSL on Aggregation"], loc='lower left')
    #ax.legend(["Ture-DP-LSL", "True-Aggregated DP-LSL"], loc='upper right')
    plt.savefig(result_path + '/' + str(num_sub_samples) + '_' + str(sub_sample_size) + '.png')
    plt.figure(1)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_traj_length", default=200, type=int)  # max length of a trajectory
    parser.add_argument("--seed", type=int)  # Sets  Numpy seeds
    parser.add_argument("--max_timesteps", default=1e6,
                        type=int)  # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--min_number_traj", default=100, type=int)
    parser.add_argument("--num_rounds", default=5, type=int)
    parser.add_argument("--batch_size_coef", default=0.5, type=float)

    parser.add_argument("--run_SA_DPLSW", action="store_true")   # If true, runs SA_DPLSW
    parser.add_argument("--run_SA_DPLSL", action="store_true")   # If true, runs SA_DPLSL
    parser.add_argument("--run_lambda_Exp", action="store_true") # If true, runs Lambda_exp
    parser.add_argument("--run_sub_sample_size_exp", action="store_true")

    parser.add_argument("--experiment_batch_lenghts", nargs='+', default=[5], type=int)
    parser.add_argument("--reg_coefs", nargs='*', default=[0.1, 1, 10, 100, 1000, 10000])
    parser.add_argument("--pow_exp", nargs='*', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--means", nargs='*', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--sigmas", nargs='*', default=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
    parser.add_argument("--values_epsilon", nargs='*', default=[0.1], type=float)

    parser.add_argument("--epsilon", default=0.1, type=float)  # privacy parameters
    parser.add_argument("--delta", default=0.1, type=float)  # privacy parameters
    parser.add_argument("--delta_prime", default=0.001, type=float)  # privacy parameters
    parser.add_argument("--batch_size", default=10000, type=int)  # privacy parameters

    parser.add_argument("--aggregationFactor", default=1, type=int)  # sub-sample and aggregate parameters
    parser.add_argument("--lambdaCoef", nargs='+', default=[10000])  # mini batch coefficient
    parser.add_argument("--number_of_subsamples", default=10)  # number of sub-samples for SA framework
    parser.add_argument("--distUB", default=10)

    parser.add_argument("--initDist", help="the initial distribution to genetate Markov chains", default="uniform")
    parser.add_argument("--numState", default=40, type=int)  # MDP parameters
    parser.add_argument("--numAbsorbingStates", default=1, type=int)  # MDP parameters
    parser.add_argument("--gamma_factor", default=0.99, type=float)  # MDP parameters
    parser.add_argument("--maxReward", default=1, type=float)  # MDP parameters
    parser.add_argument("--numGoalstates", default=1, type=int)  # MDP parameters

    args = parser.parse_args()

    result_path = ""

    print("---------------------------------------")
    if args.run_SA_DPLSW:
        result_path = f"SA_dplsw_{args.experiment_batch_lenghts}_{args.epsilon}_{args.delta}"
        print(f"Setting: running DP-LSW, batch_length: {args.experiment_batch_lenghts}, epsilon: {args.epsilon}, "
              f" delta: {args.delta}")
    elif args.run_SA_DPLSL:
        result_path = f"SA_dplsl_{args.experiment_batch_lenghts}_{args.epsilon}_{args.delta}"
        print(f"Setting: running DP-LSL, batch_length: {args.experiment_batch_lenghts}, epsilon: {args.epsilon}, "
              f" delta: {args.delta}")
    elif args.run_sub_sample_size_exp:
        print("running sub-sample size experiments")
    else:
        exit()
    print("---------------------------------------")

    if not os.path.exists(os.path.expanduser('~') + '/chap4experiments/'+ result_path):
        os.makedirs(os.path.expanduser('~') + '/chap4experiments/'+ result_path)

    result_path = str(os.path.expanduser('~') + '/chap4experiments/' + result_path)

    #######################MDP Parameters and Experiment setup###############################
    # if the absorbing state is anything except 39 (goal-state) the trajectory will not terminate

    absorbingStates = [args.numState - 1]
    goalStates = [args.numState - 2]
    stateSpace = numpy.ones(args.numState)
    for i in range(args.numState):
        stateSpace[i] = i
    stateSpace = numpy.reshape(stateSpace, (args.numState, 1))

    ##############################Privacy Parameters###############################

    ##############################MCPE Parameters##################################
    lambdaClass = 'L'
    policy = "uniform"
    distUB = args.distUB
    #####################Generating the feature matrix#############################

    exps = []

    for k in range(len(args.experiment_batch_lenghts)):
        exps.append(experiment(args.aggregationFactor, stateSpace, args.epsilon, args.delta,
                               lambdaClass, args.num_rounds, args.experiment_batch_lenghts[k], policy))

    # if args.aggregationFactor == 1 then we will have tabular setting
    featureMatrix = exps[0].featureProducer(args.aggregationFactor, stateSpace)
    dim: int = len(featureMatrix.T)

    # Starting the MC-Chain construction
    myMDP = MChain(stateSpace, exps[0].TransitionFunction, exps[0].rewardfunc, goalStates, absorbingStates, args.
                   gamma_factor, args.maxReward)
    myMCPE = MCPE(myMDP, exps[len(args.experiment_batch_lenghts) - 1].getPhi(), exps[len(args.experiment_batch_lenghts)
                                                                                     - 1].getPolicy(), "N")
    # Weight vector is used for averaging
    weightVector = []
    for i in range(args.numState):
        if i == absorbingStates[0]:
            weightVector.append(0)
        else:
            weightVector.append(1 / (args.numState - args.numAbsorbingStates))
    max_batch_size = sum(1 for line in open("newBatch.txt"))
    if args.run_SA_DPLSW:
        run_lsw_sub_sample_aggregate_experiment(result_path, exps, myMCPE, args, myMDP, max_batch_size)
    elif args.run_SA_DPLSL:
        run_lsl_sub_samp_agg_experiment(args, result_path, exps, myMCPE,  myMDP, max_batch_size)
    elif args.run_sub_sample_size_exp:
        run_sub_sample_size_experiment(exps[0], args)
    else:
        run_lambda_experiment_lsl(exps, args, myMDP)

    # weightVector = numpy.reshape(weightVector,(args.numState,1))
    # run_newGS_LSL_experiments(exps, args, args, myMDP)
    # run_newGS_LSL_vs_SmoothLSL_experiments(exps, args, args, myMDP)
    # run_lambdaExperiment_GS_LSL(myMCPE, exps,args, args, myMDP)
    # run_sa_lsw_num_sub_samples_experiment(args, exps, myMCPE, args, args, myMDP)
    # run_SubSampleAggregtate_LSL_LambdaExperiment(exps, myMDP, args,myMCPE,feature_matrix)
    # print(myMCPE.computeLambdas(myMDP, feature_matrix, args.reg_coefs, 1000, args.pow_exp)[26])
    # run_lstdExperiment(args, args, myMDP, 0.5)
