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
                                            delta_star, delta_prime, Phi, subSampleSize, result_path):
        myMCPE = MCPE(mdp, self.__Phi, self.__policy)
        V = myMCPE.realV(mdp)
        rho = mdp.startStateDistribution()

        epsilon = math.log(0.5 + math.sqrt(0.25 + (batchSize * epsilon_star) / (
                    subSampleSize * (math.sqrt(8 * number_of_sub_samples * math.log(1 / delta_prime))))))
        delta = (batchSize * (delta_star - delta_prime) / (subSampleSize * number_of_sub_samples)) * \
                (1 / (0.5 + math.sqrt(0.25 + (batchSize * epsilon_star) / (
                        subSampleSize * (math.sqrt(8 * number_of_sub_samples * math.log(1 / delta_prime)))))))

        resultsDPSA = []
        resultsSA = []
        temFVMC = []
        DPLSW_result = []
        tempMCPE = [0, 0]
        for k in range(self.__numrounds):
            # print("round"+str(k)+"has started")
            if (self.__batch_gen_param_trigger == "Y"):
                sample_batch = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)
            else:
                sample_batch = myMCPE.batchCutoff("newBatch.txt", batchSize)
            first_visit_monte_carlo = myMCPE.FVMCPE(mdp, self.__Phi, sample_batch)
            DPLSW_result.append(numpy.mat(Phi) * numpy.mat(
                myMCPE.DPLSW(first_visit_monte_carlo[0], first_visit_monte_carlo[1], mdp, self.__Phi, mdp.getGamma(),
                             epsilon_star, delta_star, batchSize)[
                    0]).T)
            tempMCPE = myMCPE.lsw_sub_sample_aggregate(sample_batch, number_of_sub_samples, mdp, self.getPhi(), epsilon,
                                                       delta, epsilon_star, delta_star, subSampleSize)
            resultsDPSA.append(tempMCPE[0])
            resultsSA.append(tempMCPE[1])
            temFVMC.append(numpy.mat(Phi) * numpy.mat(first_visit_monte_carlo[0]))

        return [resultsDPSA, resultsSA, temFVMC, V, DPLSW_result]

    def LSL_subSampleAggregateExperiment(self, mdp, regCoef, batchSize, pow_exp, maxTrajectoryLenghth,
                                         numberOfsubSamples, s, epsilon, delta, Phi, distUB, subSampleSize):
        myMCPE = MCPE(mdp, self.__Phi, self.__policy)
        V = myMCPE.realV(mdp)
        rho = mdp.startStateDistribution()
        # ridgeParam=myMCPE.computeLambdas(mdp, self.__Phi, [regCoef], batchSize, pow_exp[0])
        resultsDPSA = []
        resultsSA = []
        temLSL = []
        tempDPLSL = []
        for k in range(self.__numrounds):
            if ( self.__batch_gen_param_trigger == "Y" ):
                S = myMCPE.batchGen(mdp, maxTrajectoryLenghth, batchSize, mdp.getGamma(), self.__policy, rho)
            else:
                S = myMCPE.batchCutoff("newBatch.txt", batchSize)
            FVMC = myMCPE.FVMCPE(mdp, self.__Phi, S)
            # ridgeParam=myMCPE.computeLambdas(mdp, self.__Phi, regCoef, len(S), pow_exp)
            lsl_reidge = 10000 * math.pow(len(S), 0.4)
            LSL_result = myMCPE.LSL(FVMC[2], mdp, self.__Phi, lsl_reidge, len(S), FVMC[1])
            DPLSL_result = \
            myMCPE.DPLSL(LSL_result, FVMC[1], mdp, self.__Phi, mdp.getGamma(), epsilon, delta, lsl_reidge, len(S), rho)[
                0]
            # print('LSL Norm: '+str(numpy.linalg.norm(LSL_result)))
            # print('DPLSL Norm: '+str(numpy.linalg.norm(DPLSL_result)))
            tempSA = myMCPE.lsl_sub_sample_aggregate(S, s, numberOfsubSamples, mdp, self.getPhi(), regCoef, pow_exp,
                                                     batchSize, epsilon, delta, distUB, subSampleSize)
            resultsDPSA.append(tempSA[0])
            resultsSA.append(tempSA[1])
            temLSL.append(numpy.mat(Phi) * numpy.mat(LSL_result))
            tempDPLSL.append(numpy.mat(Phi) * numpy.mat(DPLSL_result))
        return [resultsDPSA, resultsSA, temLSL, V, tempDPLSL]

    def rewardfunc(self, destState, goalstates, maxReward):
        if destState in goalstates:
            return maxReward
        else:
            return 0


def run_lambdaExperiment_LSL(experimentList, myMDP_Params, myExp_Params, myMDP):
    i = 0
    expResults = []
    for i in range(len(myExp_Params.experiment_batch_lenghts)):
        expResults.append(experimentList[i].lambdaExperiment_LSL(myMDP, myExp_Params.experiment_batch_lenghts[i],
                                                                 myExp_Params.maxTrajLength, myExp_Params.reg_coefs,
                                                                 myExp_Params.pow_exp))
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    # ax.set_xscale('log')
    ax.set_yscale('log')
    num_reidge_params = len(myExp_Params.reg_coefs) * len(myExp_Params.pow_exp)

    # Real_vs_LSL_list=numpy.zeros(len(num_reidge_params))
    # Real_vs_DPLSL_list=numpy.zeros(len(num_reidge_params))
    expReal_vs_LS = numpy.zeros((len(myExp_Params.experiment_batch_lenghts), num_reidge_params))
    expReal_vs_DPLSL = numpy.zeros((len(myExp_Params.experiment_batch_lenghts), num_reidge_params))
    reidgeParamLsit = []
    i = 0
    num_reidge_params = len(myExp_Params.reg_coefs) * len(myExp_Params.pow_exp)
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


def run_SALSW_numSubs_experimet(experimentList, myMCPE, myMDP_Params, myExp_Params, myMDP, subSampleSize):
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


def run_lsw_sub_sample_aggregate_experiment(result_path, experiment_list, myMCPE, args, myMDP):
    exp_results_dpsa = []
    exp_results_sa = []
    exp_results_lsw = []
    exp_results_v = []
    exp_results_dplsw = []

    number_of_sub_samples_exponent = 8.0 / 3.0
    sub_sample_size_exponent = 0.75

    # Note that as theory suggests number_of_sub_samples_exponent * sub_sample_size_exponent =2

    for i in range(len(args.experiment_batch_lenghts)):
        number_of_sub_samples = math.floor(math.pow(args.experiment_batch_lenghts[i], number_of_sub_samples_exponent))
        subSampleSize = math.floor(math.pow(args.experiment_batch_lenghts[i], sub_sample_size_exponent))
        tempSAE = experiment_list[i].lsw_sub_sample_aggregate_experiment(myMDP, args.experiment_batch_lenghts[i],
                                                                         args.max_traj_length, number_of_sub_samples,
                                                                         args.epsilon, args.delta, args.delta_prime,
                                                                         experiment_list[0].getPhi(), subSampleSize,
                                                                         result_path)
        with open(f"./{result_path}/{args.experiment_batch_lenghts[i]}_{number_of_sub_samples}_{subSampleSize}.npy",
                  'wb') as f:
            numpy.save(f, tempSAE[0])
            numpy.save(f, tempSAE[1])
            numpy.save(f, tempSAE[2])
            numpy.save(f, tempSAE[3])
            numpy.save(f, tempSAE[4])

        exp_results_dpsa.append(tempSAE[0])
        exp_results_sa.append(tempSAE[1])
        exp_results_lsw.append(tempSAE[2])
        exp_results_v.append(tempSAE[3])
        exp_results_dplsw.append(tempSAE[4])

    ax = plt.gca()
    ax.set_prop_cycle(color=['red', 'green', 'blue', 'purple'])

    mean_v_vs_dpsa = numpy.zeros(len(args.experiment_batch_lenghts))
    std_v_vs_dpsa = numpy.zeros(len(args.experiment_batch_lenghts))
    v_vs_dpsa_bldu = numpy.zeros(len(args.experiment_batch_lenghts))
    v_vs_dpsa_bldl = numpy.zeros(len(args.experiment_batch_lenghts))
    v_vs_dpsa_blm = numpy.zeros(len(args.experiment_batch_lenghts))
    mean_v_vs_sa = numpy.zeros(len(args.experiment_batch_lenghts))
    std_V_vs_SA = numpy.zeros(len(args.experiment_batch_lenghts))
    V_vs_SA_bldu = numpy.zeros(len(args.experiment_batch_lenghts))
    V_vs_SA_bldl = numpy.zeros(len(args.experiment_batch_lenghts))
    V_vs_SA_blm = numpy.zeros(len(args.experiment_batch_lenghts))
    mean_V_vs_LSW = numpy.zeros(len(args.experiment_batch_lenghts))
    std_V_vs_LSW = numpy.zeros(len(args.experiment_batch_lenghts))
    V_vs_LSW_bldu = numpy.zeros(len(args.experiment_batch_lenghts))
    V_vs_LSW_bldl = numpy.zeros(len(args.experiment_batch_lenghts))
    V_vs_LSW_blm = numpy.zeros(len(args.experiment_batch_lenghts))
    mean_V_vs_DPLSW = numpy.zeros(len(args.experiment_batch_lenghts))
    std_V_vs_DPLSW = numpy.zeros(len(args.experiment_batch_lenghts))
    V_vs_DPLSW_bldu = numpy.zeros(len(args.experiment_batch_lenghts))
    V_vs_DPLSW_bldl = numpy.zeros(len(args.experiment_batch_lenghts))
    V_vs_DPLSW_blm = numpy.zeros(len(args.experiment_batch_lenghts))

    dim = len(experiment_list[0].getPhi())

    for j in range(len(args.experiment_batch_lenghts)):
        tempDPSA = [[] for x in range(len(args.experiment_batch_lenghts))]
        tempSA = [[] for x in range(len(args.experiment_batch_lenghts))]
        tempV = numpy.reshape(exp_results_v[j], (len(experiment_list[i].getPhi()), 1))
        tempLSW = [[] for x in range(len(args.experiment_batch_lenghts))]
        tempDPLSW = [[] for x in range(len(args.experiment_batch_lenghts))]

        for k in range(args.num_rounds):
            tempDPSA[j].append(
                myMCPE.weighted_dif_L2_norm(myMDP, tempV, numpy.reshape(exp_results_dpsa[j][k], (dim, 1))))
            tempSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, numpy.reshape(exp_results_sa[j][k], (dim, 1))))
            vhat = numpy.reshape(exp_results_lsw[j][k], (dim, 1))
            tempLSW[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhat))
            vhatDPLSW = numpy.reshape(exp_results_dplsw[j][k], (dim, 1))
            tempDPLSW[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhatDPLSW))

        new_temp = tempLSW[j]
        mean_V_vs_LSW[j] = abs(numpy.average(new_temp))
        std_V_vs_LSW[j] = numpy.std(new_temp)
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

        mean_v_vs_dpsa[j] = numpy.average(tempDPSA[j])
        std_v_vs_dpsa[j] = numpy.std(tempDPSA[j])  # bld
        v_vs_dpsa_bldu[j] = math.log10(abs(mean_v_vs_dpsa[j] + std_v_vs_dpsa[j])) - math.log10(abs(mean_v_vs_dpsa[j]))
        v_vs_dpsa_bldl[j] = -math.log10(abs(mean_v_vs_dpsa[j] - std_v_vs_dpsa[j])) + math.log10(abs(mean_v_vs_dpsa[j]))
        v_vs_dpsa_blm[j] = math.log10(abs(mean_v_vs_dpsa[j]))

        mean_v_vs_sa[j] = numpy.average(tempSA[j])
        std_V_vs_SA[j] = numpy.std(tempSA[j])  # bld
        V_vs_SA_bldu[j] = math.log10((mean_v_vs_sa[j] + std_V_vs_SA[j])) - math.log10((mean_v_vs_sa[j]))
        V_vs_SA_bldl[j] = -math.log10((mean_v_vs_sa[j] - std_V_vs_SA[j])) + math.log10((mean_v_vs_sa[j]))
        V_vs_SA_blm[j] = math.log10((mean_v_vs_sa[j]))

    ax.errorbar(args.experiment_batch_lenghts, V_vs_LSW_blm,  yerr=[V_vs_LSW_bldu, V_vs_LSW_bldl])
    ax.errorbar(args.experiment_batch_lenghts, V_vs_DPLSW_blm, yerr=[V_vs_DPLSW_bldu, V_vs_DPLSW_bldl])
    ax.errorbar(args.experiment_batch_lenghts, V_vs_SA_blm,  yerr=[V_vs_SA_bldu, V_vs_SA_bldl])
    ax.errorbar(args.experiment_batch_lenghts, v_vs_dpsa_blm, yerr=[v_vs_dpsa_bldu, v_vs_dpsa_bldl])

    rmse_results = [V_vs_LSW_blm, V_vs_DPLSW_blm, V_vs_SA_blm, v_vs_dpsa_blm]

    with open(f"./{result_path}/results_{args.experiment_batch_lenghts}_{number_of_sub_samples}_{subSampleSize}_"
              f"{args.epsilon}_{args.delta}.csv", 'a') as csvfile:
        writer = csv.writer(csvfile)
        fieldnames = ['Lower Bound', 'Mean', 'Upper Bound']
        writer.writerow(fieldnames)
        # for per in range(2):
        for i in range(len(args.experiment_batch_lenghts)):
            writer.writerow([V_vs_DPLSW_bldl[i], V_vs_DPLSW_blm[i], V_vs_DPLSW_bldu[i]])
            writer.writerow([v_vs_dpsa_bldl[i], v_vs_dpsa_blm[i], v_vs_dpsa_bldu[i]])
            writer.writerow([V_vs_LSW_bldl[i], V_vs_LSW_blm[i], V_vs_LSW_bldu[i]])
            writer.writerow([V_vs_SA_bldl[i], V_vs_SA_blm[i], V_vs_SA_bldu[i]])

    ax.set_xscale('log')
    plt.ylabel('(log) RMSE)')
    plt.xlabel('(log) Batch Size')
    plt.legend(["LSW-Real", "DPLSW-Real", "(LSW)SA-Real", "(LSW)DPSA-Real"], loc='upper right')
    #plt.legend(["DPLSW vs. True", "SA-DPLSW vs. True"], loc=1)
    plt.title("epsilon= " + str(args.epsilon) + ", delta= " + str(args.delta) + ", number of sub samples: \sqrt(m)")
    ax.plot(args.experiment_batch_lenghts, rmse_results[0])
    ax.plot(args.experiment_batch_lenghts, rmse_results[1])
    ax.plot(args.experiment_batch_lenghts, rmse_results[2])
    ax.plot(args.experiment_batch_lenghts, rmse_results[3])
    # ax.plot(args.experiment_batch_lenghts,realV_vs_FVMC)
    # ax.plot(args.experiment_batch_lenghts,LSL_vs_DPLSL)
    plt.savefig(f"./{result_path}/results_{args.experiment_batch_lenghts}_{number_of_sub_samples}_{subSampleSize}_"
                f"{args.epsilon}_{args.delta}.png")
    plt.show()


def run_SubSampleAggregtate_LSL_LambdaExperiment(experimentList, myMDP, myExp_Params, myMCP, Phi):
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


def run_LSL_SubSampAggExperiment(experimentList, myMCPE, myMDP_Params, myExp_Params, myMDP):
    expResultsDPSA = []
    expResultsSA = []
    expResultsLSL = []
    expResultsV = []
    expResultsDPLSL = []

    # Note that as theory suggests numberOfsubSamples_Exponent * subSampleSize_exponent = 2

    numberOfsubSamples_Exponent = 8.0 / 3.0
    subSampleSize_exponent = 0.75

    for i in range(len(myExp_Params.experiment_batch_lenghts)):
        numberOfsubSamples = math.floor(math.pow(myExp_Params.experiment_batch_lenghts[i], numberOfsubSamples_Exponent))
        subSampleSize = math.floor(math.pow(myExp_Params.experiment_batch_lenghts[i], subSampleSize_exponent))
        s = int(numpy.sqrt(numberOfsubSamples))
        tempSAE = experimentList[i].LSL_subSampleAggregateExperiment(myMDP, myExp_Params.lambdaCoef,
                                                                     myExp_Params.experiment_batch_lenghts[i],
                                                                     myExp_Params.pow_exp, myExp_Params.maxTrajLength,
                                                                     numberOfsubSamples, s, myExp_Params.epsilon,
                                                                     myExp_Params.delta, experimentList[0].getPhi(),
                                                                     myExp_Params.distUB, subSampleSize)
        expResultsDPSA.append(tempSAE[0])
        expResultsSA.append(tempSAE[1])
        expResultsLSL.append(tempSAE[2])
        expResultsV.append(tempSAE[3])
        expResultsDPLSL.append(tempSAE[4])

    ax1 = plt.gca()
    ax1.set_prop_cycle(color=['red', 'green', 'blue'])
    # ax1.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    # ax.set_color_cycle(['b', 'r', 'g', 'y', 'k', 'c', 'm'])

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
    mean_V_vs_LSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_V_vs_LSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_LSL_bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_LSL_bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_LSL_blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))

    mean_V_vs_DPLSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    std_V_vs_DPLSL = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_DPLSL_bldu = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_DPLSL_bldl = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))
    V_vs_DPLSL_blm = numpy.zeros(len(myExp_Params.experiment_batch_lenghts))

    dim = len(experimentList[0].getPhi())

    for j in range(len(myExp_Params.experiment_batch_lenghts)):
        tempDPSA = [[] for x in range(len(myExp_Params.experiment_batch_lenghts))]
        tempSA = [[] for x in range(len(myExp_Params.experiment_batch_lenghts))]
        tempV = numpy.reshape(expResultsV[j], (myMDP_Params.numState, 1))
        tempLSL = [[] for x in range(len(myExp_Params.experiment_batch_lenghts))]
        tempDPLSL = [[] for x in range(len(myExp_Params.experiment_batch_lenghts))]
        for k in range(myExp_Params.numRounds):
            tempDPSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, numpy.reshape(expResultsDPSA[j][k], (dim, 1))))
            tempSA[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, numpy.reshape(expResultsSA[j][k], (dim, 1))))
            vhat = numpy.reshape(expResultsLSL[j][k], (dim, 1))
            tempLSL[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhat))
            vhatDPLSL = numpy.reshape(expResultsDPLSL[j][k], (dim, 1))
            tempDPLSL[j].append(myMCPE.weighted_dif_L2_norm(myMDP, tempV, vhatDPLSL))
        # tempDPSA=tempDPSA/args.numRounds
        # tempSA=tempSA/args.numRounds
        # tempLSW=tempLSW/args.numRounds

        # mean_V_vs_LSW[j]=numpy.average(tempLSW-tempV)
        # std_V_vs_LSW[j] = numpy.std(tempLSW-tempV)#bld
        # V_vs_LSW_bldu[j] = math.log10(abs(mean_V_vs_LSW[j]+std_V_vs_LSW[j]))-math.log10(abs(mean_V_vs_LSW[j]))
        # V_vs_LSW_bldl[j] = -math.log10(abs(mean_V_vs_LSW[j]-std_V_vs_LSW[j]))+math.log10(abs(mean_V_vs_LSW[j]))
        # V_vs_LSW_blm[j] = math.log10(abs(mean_V_vs_LSW[j]))
        mean_V_vs_LSL[j] = abs(numpy.average(tempLSL[j]))
        std_V_vs_LSL[j] = numpy.std(tempLSL[j])
        V_vs_LSL_bldu[j] = math.log10(abs(mean_V_vs_LSL[j] + std_V_vs_LSL[j])) - math.log10(abs(mean_V_vs_LSL[j]))
        V_vs_LSL_bldl[j] = (-math.log10(abs(mean_V_vs_LSL[j] - std_V_vs_LSL[j])) + math.log10(abs(mean_V_vs_LSL[j])))
        V_vs_LSL_blm[j] = math.log10(abs(mean_V_vs_LSL[j]))

        mean_V_vs_DPLSL[j] = numpy.average(tempDPLSL[j])
        std_V_vs_DPLSL[j] = numpy.std(tempDPLSL[j])  # bld
        V_vs_DPLSL_bldu[j] = math.log10(abs(mean_V_vs_DPLSL[j] + std_V_vs_DPLSL[j])) - math.log10(
            abs(mean_V_vs_DPLSL[j]))
        V_vs_DPLSL_bldl[j] = (
                    -math.log10(abs(mean_V_vs_DPLSL[j] - std_V_vs_DPLSL[j])) + math.log10(abs(mean_V_vs_DPLSL[j])))
        V_vs_DPLSL_blm[j] = math.log10(abs(mean_V_vs_DPLSL[j]))

        mean_V_vs_DPSA[j] = numpy.average(tempDPSA[j])
        std_V_vs_DPSA[j] = numpy.std(tempDPSA[j])  # bld
        V_vs_DPSA_bldu[j] = math.log10(abs(mean_V_vs_DPSA[j] + std_V_vs_DPSA[j])) - math.log10(abs(mean_V_vs_DPSA[j]))
        V_vs_DPSA_bldl[j] = (
                    -math.log10(abs(mean_V_vs_DPSA[j] - std_V_vs_DPSA[j])) + math.log10(abs(mean_V_vs_DPSA[j])))
        V_vs_DPSA_blm[j] = math.log10(abs(mean_V_vs_DPSA[j]))

        mean_V_vs_SA[j] = numpy.average(tempSA[j])
        std_V_vs_SA[j] = numpy.std(tempSA[j])  # bld
        V_vs_SA_bldu[j] = math.log10((mean_V_vs_SA[j] + std_V_vs_SA[j])) - math.log10((mean_V_vs_SA[j]))
        V_vs_SA_bldl[j] = (-math.log10((mean_V_vs_SA[j] - std_V_vs_SA[j])) + math.log10((mean_V_vs_SA[j])))
        V_vs_SA_blm[j] = math.log10((mean_V_vs_SA[j]))

        # =======================================================================
        # mean_V_vs_DPLSL[j]=numpy.average(tempDPLSL[j])
        # std_V_vs_DPLSL[j] = numpy.std(tempDPLSL[j])
        # V_vs_DPLSL_bldu[j] = math.log10((mean_V_vs_DPLSL[j]+std_V_vs_DPLSL[j]))
        # V_vs_DPLSL_bldl[j] = math.log10(abs(mean_V_vs_DPLSL[j]-std_V_vs_DPLSL[j]))
        # V_vs_DPLSL_blm[j] =math.log10(abs(mean_V_vs_DPLSL[j]))
        # =======================================================================

    ax1.set_xscale('log')
    ax1.errorbar(myExp_Params.experiment_batch_lenghts, V_vs_LSL_blm, yerr=[V_vs_LSL_bldu, V_vs_LSL_bldl])
    ax1.legend(["LSL-Real"], loc=1)
    ax1.set_xlabel('(log)Batch Size')
    ax1.set_ylabel('(log) RMSE')
    ax1.set_title("epsilon= " + str(myExp_Params.epsilon) + ", delta= " + str(
        myExp_Params.delta) + ", number of sub samples: \sqrt(m)" + "  lambda= " + "  lambda= " + " 10000m^0.4")
    ax1.plot()

    ax2.set_xscale('log')
    ax2.errorbar(myExp_Params.experiment_batch_lenghts, V_vs_DPLSL_blm, yerr=[V_vs_DPLSL_bldu, V_vs_DPLSL_bldl])
    ax2.legend(["DPLSL-Real"], loc=1)
    ax2.set_xlabel('(log)Batch Size')
    ax2.set_ylabel('(log) RMSE')
    ax2.set_title("epsilon= " + str(myExp_Params.epsilon) + ", delta= " + str(
        myExp_Params.delta) + ", number of sub samples: \sqrt(m)" + "  lambda= " + " 10000m^0.4")
    ax2.plot()

    ax3.set_xscale('log')
    ax3.errorbar(myExp_Params.experiment_batch_lenghts, V_vs_SA_blm, yerr=[V_vs_SA_bldu, V_vs_SA_bldl])
    ax3.legend(["(LSL)SA-Real"], loc=1)
    ax3.set_xlabel('(log)Batch Size')
    ax3.set_ylabel('(log) RMSE')
    ax3.set_title("epsilon= " + str(myExp_Params.epsilon) + ", delta= " + str(
        myExp_Params.delta) + ", number of sub samples: \sqrt(m)" + "  lambda= " + "  lambda= " + "100m^0.5")
    ax3.plot()

    ax4.set_xscale('log')
    ax4.errorbar(myExp_Params.experiment_batch_lenghts, V_vs_DPSA_blm, yerr=[V_vs_DPSA_bldu, V_vs_DPSA_bldl])
    ax4.legend(["DPSA(LSL)-Real"], loc=1)
    ax4.set_xlabel('(log)Batch Size')
    ax4.set_ylabel('(log) RMSE')
    ax4.set_title("epsilon= " + str(myExp_Params.epsilon) + ", delta= " + str(
        myExp_Params.delta) + ", number of sub samples: \sqrt(m)" + "  lambda= " + "100m^0.5")
    ax4.plot()
    # plt.ylabel('l2-Norm')
    # plt.xlabel('(log)Batch Size')
    #     ax1.set_xlabel('(log)Batch Size')
    #     ax1.set_ylabel('(log) RMSE')
    #     ax1.set_title("epsilon= "+str(args.epsilon)+", delta= "+str(args.delta)+", number of sub samples: \sqrt(m)"+"  lambda= "+str(args.lambdaCoef)+"m^"+str(args.pow_exp[0]))
    #     plt.legend(["LSL-Real", "DPLSL-Real",  "SA(LSL)-Real","DPSA(LSL)-Real"],loc=1)
    #     ax1.set_xlim([-5, 5])
    #     plt.title("epsilon= "+str(args.epsilon)+", delta= "+str(args.delta)+", number of sub samples: \sqrt(m)"+"  lambda= "+str(args.lambdaCoef)+"m^"+str(args.pow_exp[0]))
    #     ax.plot(args.experiment_batch_lenghts,realV_vs_FVMC)
    #     ax.plot(args.experiment_batch_lenghts,LSL_vs_DPLSL)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_traj_length", default=200, type=int)  # max length of a trajectory
    parser.add_argument("--seed", type=int)  # Sets  Numpy seeds
    parser.add_argument("--max_timesteps", default=1e6,
                        type=int)  # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--min_number_traj", default=100, type=int)
    parser.add_argument("--num_rounds", default=5, type=int)

    parser.add_argument("--run_SA_DPLSW", action="store_true")   # If true, runs SA_DPLSW
    parser.add_argument("--run_SA_DPLSL", action="store_true")   # If true, runs SA_DPLSL
    parser.add_argument("--run_lambda_Exp", action="store_true") # If true, runs Lambda_exp

    parser.add_argument("--experiment_batch_lenghts", nargs='*', default=[5, 10, 15, 30])
    parser.add_argument("--reg_coefs", nargs='*', default=[0.1, 1, 10, 100, 1000, 10000])
    parser.add_argument("--pow_exp", nargs='*', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--means", nargs='*', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--sigmas", nargs='*', default=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])

    parser.add_argument("--epsilon", default=0.1)  # privacy parameters
    parser.add_argument("--delta", default=0.15)  # privacy parameters
    parser.add_argument("--delta_prime", default=0.1)  # privacy parameters

    parser.add_argument("--aggregationFactor", default=1)  # sub-sample and aggregate parameters
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
    else:
        exit()
    print("---------------------------------------")

    if not os.path.exists(f"./{result_path}/results"):
        os.makedirs(f"./{result_path}/results")

    result_path = f"./{result_path}/results"

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

    run_lsw_sub_sample_aggregate_experiment(result_path, exps, myMCPE, args, myMDP)
    # weightVector = numpy.reshape(weightVector,(args.numState,1))
    # run_lambdaExperiment_LSL(exps, args, args, myMDP)
    # run_newGS_LSL_experiments(exps, args, args, myMDP)
    # run_newGS_LSL_vs_SmoothLSL_experiments(exps, args, args, myMDP)
    # run_lambdaExperiment_GS_LSL(myMCPE, exps,args, args, myMDP)
    # run_SALSW_numSubs_experimet(exps, myMCPE, args, args, myMDP)
    #run_LSL_SubSampAggExperiment(exps, myMCPE, args, args, myMDP)
    # run_SubSampleAggregtate_LSL_LambdaExperiment(exps, myMDP, args,myMCPE,featureMatrix)
    # print(myMCPE.computeLambdas(myMDP, featureMatrix, args.reg_coefs, 1000, args.pow_exp)[26])
    # run_lstdExperiment(args, args, myMDP, 0.5)
