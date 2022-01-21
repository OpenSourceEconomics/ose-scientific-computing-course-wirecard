from numpy import *
import pandas as pd
import random
import nlopt
import numpy as np
import matplotlib.pyplot as plt
import numbers
import math
import random
import autograd.numpy as ag
from autograd import grad
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.function_base import vectorize
from autograd import value_and_grad
np.set_printoptions(precision=20)
pd.set_option("display.precision", 14)





###### The function that performs the optimization routine for nlopt algorithms





def minimization_guvenen_4(f,computational_budget,algo,x_0,n,problem_info,x_tol_abs,f_tol_abs,local_tolerance_x=None,local_tolerance_f=None):
    
    
    
    
    ##################################     Input that need to be specified ################################
    #######################################################################################################
    
    
    ## x_tol: is the absolute Tolerance allowed
    ## f_tol: tolerance in function value allowed
    ## f: we need to specify the objective function
    ## computational budget: is a vector that contains different computational budgets between 0 and 10^5
    ## algo: specify the algorithm you want to use from the nlopt library -> argument has to have the form:
    ######## nlopt.algorithm_name e.g. nlopt.GN_ISRES for ISRES Algorithm
    ## algorithm: specify the algorithm 
    ## x_0: contains the randomly generated starting points -> pandas dataframe containing starting values
    ## n: number of dimensions the problem should have
    ## problem_info: object that that contains known information about the objective function 
                    ## as for example the domain
                    ## the best solver
                    ## function value of the best solver etc
    ### If you want to stop the optimization routine when the x_tol_abs for convergence is met 
    ########   -> plug in -inf for f_tol_abs
    ##### If you want to stop the optimization routine when the f_tol_abs convergence criterion is met specify:
    ######## -> x_tol_abs=-inf
    
    ######################################      Output       ################################################
    
    #### returns a dataframe containing:
    #### a vector of the optimizer -> first n columns # coordinate vector
    #### the function value of the optimizer -> next column                ##### this is done 100 times
    #### number of function evaluations -> next columns                      #### for all 100 starting points
    ### accuracy measures as specified in Guvenen et al. 
    
    #np.set_printoptions(precision=20)
    
    fwrapped=lambda x,grad:f(x)
    
    if algo==19:
        
        global_optimum=nlopt.opt(nlopt.GN_CRS2_LM,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer=global_optimum.optimize(np.array(x_0.iloc[i]))
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    elif algo==42:
        
        global_optimum=nlopt.opt(nlopt.GN_ESCH,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer=global_optimum.optimize(np.array(x_0.iloc[i]))
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    
    
    elif algo==35:
        
        global_optimum=nlopt.opt(nlopt.GN_ISRES,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer=global_optimum.optimize(np.array(x_0.iloc[i]))
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    
    
    
    
    elif algo==20:
        
        global_optimum=nlopt.opt(nlopt.GN_MLSL,n)
        local_opt=nlopt.opt(nlopt.LN_NELDERMEAD,10)
        local_opt.set_xtol_abs(local_tolerance_x)
        local_opt.set_ftol_abs(local_tolerance_f)
        global_optimum.set_local_optimizer(local_opt)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer=global_optimum.optimize(np.array(x_0.iloc[i]))
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    
    
    
    elif algo==8:
        
        global_optimum=nlopt.opt(nlopt.GD_STOGO,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(f)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer=global_optimum.optimize(np.array(x_0.iloc[i]))
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,
                                       num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    
    
    
    
    
def get_success_results_correct_x(results,tau,computational_budget):
    
    ###### first drop all Problems where f_eval > max_eval allowed
    results.comp_budget=results['comp_budget'].astype(float)
    results.f_evals=results['f_evals'].astype(float)
    
    vector_trues=np.less_equal(results.f_evals,results.comp_budget)
    
    #### create dataframe
    
    d = {'comp_budget': results.comp_budget, 'f_evals': results.f_evals,'success_crit_x': results.success_crit_x,'trues':vector_trues}
    df = pd.DataFrame(data=d)
    df_filtered=df[df['trues']==True]
    df_filtered
    results.comp_budget=results['comp_budget'].astype(int)
    comp_budge=np.array(computational_budget)
    
    df=[]
    for i in range (comp_budge.size):
        comp_b_i=computational_budget[i]
        data_i=df_filtered[df_filtered.comp_budget==comp_b_i]
        success_criterion_i=np.array(data_i.success_crit_x)
        success_crit=success_criterion_i.astype(np.float)
        comparison_vector=np.full((1,success_crit.size),tau)
        result_r=np.less(success_crit,comparison_vector)
        df_describe=pd.DataFrame(np.transpose(result_r))
        df_101=df_describe.rename(columns={0:"results"})
        trues=df_101[df_101.results==True]
        success_prob=np.array(trues.size/100)
        stats=np.hstack((comp_b_i,success_prob))
        df.append(stats)
    
    dataframe_success=pd.DataFrame(df)
    dataframe_success_1= dataframe_success.rename(columns={0:"comp_budget",1:'success_prob'})
    
    return dataframe_success_1



def success_results_x_all(results_list,tau,computational_budget,cols): #### columns argument of type columns={0:"comp_budget",1:"first algo name".....}
    #df.insert(0, 'new_column', ['a','b','c'])
    df=[]
    df.append(np.array(computational_budget))
    for i in range(len(results_list)):
        success_result_i=get_success_results_correct_x(results_list[i],tau,computational_budget)
        #success_result_i_array=np.array(success_result_i.iloc[:,1]).reshape(len(computational_budget),1)
        df.append(success_result_i.iloc[:,1])
    
    dataframe_success=pd.DataFrame(df)
    dataframe_success_1=dataframe_success.transpose()
    dataframe_success_2= dataframe_success_1.rename(columns=cols)
    
    return dataframe_success_2




def data_profile(success_results_frame,color_vector):
    plt.figure(figsize=(10,5))
    plt.xscale('log')
    for i in range(len(success_results_frame)):
        plt.plot(success_results_frame.comp_budget, success_results_frame.iloc[:,i+1], color=color_vector[i], 
                 linestyle='dashed', linewidth = 1,marker='x', markerfacecolor='blue',markersize=5)
    plt.ylim(-0.05,1.1)
    plt.xlim(0,102000)
 
    # naming the x axis
    plt.xlabel('Comp_budget')
    # naming the y axis
    plt.ylabel('Success_Rate')
 
    # giving a title to my graph
    plt.title('Data_Profile')

    # function to show the plot
    return plt.show()
    
    
    
    
    
    
    
def get_success_results_correct(results,tau,computational_budget,success_criterion):
    
    no_start_points=len(results)/len(computational_budget)
    
    if success_criterion==0:
        results.comp_budget=results['comp_budget'].astype(float)
        results.f_evals=results['f_evals'].astype(float)
    
        vector_trues=np.less_equal(results.f_evals,results.comp_budget)
    
    #### create dataframe
    
        d = {'comp_budget': results.comp_budget, 'f_evals': results.f_evals,'success_crit_x': results.success_crit_x,'trues':vector_trues}
        df = pd.DataFrame(data=d)
        df_filtered=df[df['trues']==True]
        df_filtered
        results.comp_budget=results['comp_budget'].astype(int)
        comp_budge=np.array(computational_budget)
        df=[]
        for i in range (comp_budge.size):
            comp_b_i=computational_budget[i]
            data_i=df_filtered[df_filtered.comp_budget==comp_b_i]
            success_criterion_i=np.array(data_i.success_crit_x)
            success_crit=success_criterion_i.astype(np.float)
            comparison_vector=np.full((1,success_crit.size),tau)
            result_r=np.less(success_crit,comparison_vector)
            df_describe=pd.DataFrame(np.transpose(result_r))
            df_101=df_describe.rename(columns={0:"results"})
            trues=df_101[df_101.results==True]
            success_prob=np.array(trues.size/no_start_points)
            stats=np.hstack((comp_b_i,success_prob))
            df.append(stats)
    
        dataframe_success=pd.DataFrame(df)
        dataframe_success_1= dataframe_success.rename(columns={0:"comp_budget",1:'success_prob_x'})
    
        return dataframe_success_1
    
    elif success_criterion==1:
        
        results.comp_budget=results['comp_budget'].astype(float)
        results.f_evals=results['f_evals'].astype(float)
    
        vector_trues=np.less_equal(results.f_evals,results.comp_budget)
    
    #### create dataframe
    
        d = {'comp_budget': results.comp_budget, 'f_evals': results.f_evals,'success_crit_f': results.success_crit_f,'trues':vector_trues}
        df = pd.DataFrame(data=d)
        df_filtered=df[df['trues']==True]
        df_filtered
        results.comp_budget=results['comp_budget'].astype(int)
        comp_budge=np.array(computational_budget)
        df=[]
        for i in range (comp_budge.size):
            comp_b_i=computational_budget[i]
            data_i=df_filtered[df_filtered.comp_budget==comp_b_i]
            success_criterion_i=np.array(data_i.success_crit_f)
            success_crit=success_criterion_i.astype(np.float)
            comparison_vector=np.full((1,success_crit.size),tau)
            result_r=np.less(success_crit,comparison_vector)
            df_describe=pd.DataFrame(np.transpose(result_r))
            df_101=df_describe.rename(columns={0:"results"})
            trues=df_101[df_101.results==True]
            success_prob=np.array(trues.size/no_start_points)
            stats=np.hstack((comp_b_i,success_prob))
            df.append(stats)
    
        dataframe_success=pd.DataFrame(df)
        dataframe_success_1= dataframe_success.rename(columns={0:"comp_budget",1:'success_prob_f'})
    
        return dataframe_success_1
    
    
def success_results_all(results_list,tau,computational_budget,cols,success_criterion): 
    #### columns argument of type columns={0:"comp_budget",1:"first algo name".....}
    #df.insert(0, 'new_column', ['a','b','c'])
    df=[]
    df.append(np.array(computational_budget))
    for i in range(len(results_list)):
        success_result_i=get_success_results_correct(results_list[i],tau,computational_budget,success_criterion)
        #success_result_i_array=np.array(success_result_i.iloc[:,1]).reshape(len(computational_budget),1)
        df.append(success_result_i.iloc[:,1])
    
    dataframe_success=pd.DataFrame(df)
    dataframe_success_1=dataframe_success.transpose()
    dataframe_success_2= dataframe_success_1.rename(columns=cols)
    
    return dataframe_success_2



def data_profile_general(success_results_frame,color_vector,marker_vector,algo_names):
    plt.figure(figsize=(20,10))
    plt.xscale('log')
    #success_results_frame_1=success_results_frame.drop('comp_budget', 1)
    for i in range(len(success_results_frame.columns)-1):
        plt.plot(success_results_frame.comp_budget, success_results_frame.iloc[:,i+1], color=color_vector[i], linestyle='dashed',
                 linewidth = 2,marker=marker_vector[i], markerfacecolor=color_vector[i], markersize=8,label=algo_names[i])
    plt.ylim(-0.05,1.1)
    plt.xlim(0,102000)
    plt.xscale("log")
 
    # naming the x axis
    plt.xlabel('Comp_budget')
    # naming the y axis
    plt.ylabel('Success_Rate')
 
    # giving a title to my graph
    plt.title('Data_Profile')
    plt.legend()

    # function to show the plot
    return plt.show()



###### filter function that drops all sucessfull implementations


def filter_dataframe(results,tau,success_criterion,n):
    
    cols_dev=('f_evals','comp_budget','f_val_opt','success_crit_x','success_crit_f','6','7','8',
              '9','10','11','12','13','14','15')
    
    res_filtered=results.loc[:,cols_dev[0:(n+5)]]
    
    ##### we want to select all FAILED implementations in order to plot the deviation
    ###### add true vector which indicated whether computational budget was exceeded
    res_filtered.comp_budget=res_filtered['comp_budget'].astype(float)
    res_filtered.f_evals=res_filtered['f_evals'].astype(float)
    
    vector_trues=np.less_equal(res_filtered.f_evals,res_filtered.comp_budget)
    
    
    res_filtered.loc[:,'trues_lesser_f_evals']=vector_trues
    
    if success_criterion==0:
        
        #### initialise comparison
        trues_success_crit_x=np.less(results.success_crit_x,tau)
        trues_x=np.transpose(trues_success_crit_x)
        res_filtered.loc[:,'trues_success_crit_x']=trues_x
        res_filtered_finished=res_filtered.drop(res_filtered[(res_filtered.trues_lesser_f_evals==True)
                                                             &(res_filtered.trues_success_crit_x==True)].index)
        return res_filtered_finished
    
    if success_criterion==1:
        #### initialise comparison
        trues_success_crit_f=np.less(results.success_crit_f,tau)
        trues_f=np.transpose(trues_success_crit_f)
        res_filtered.loc[:,'trues_success_crit_f']=trues_f
        res_filtered_finished=res_filtered.drop(res_filtered[(res_filtered.trues_lesser_f_evals==True)
                                                             &(res_filtered.trues_success_crit_f==True)].index)
        return res_filtered_finished
    
    
    

    
    
    
    
    
    
    
def get_deviation_results_correct(results,tau,success_criterion,computational_budget,info_object,n,computation_method_x=None):
    
    
    ### if computation_method_x=0 it computes Mean absolute error (MAE) as distance measure
    ### if computation_method_x=1 it computes the Root Mean squared error (RMSE) as distance measure which is sqrt(n)^(-1)*Euclidean Distance
    cols_coordinate=('6','7','8','9','10','11','12','13','14','15')
    
    cols=('f_evals','comp_budget','f_val_opt','success_crit_x','success_crit_f','6','7','8','9','10','11','12','13','14','15')
    ##### first get filtered data to sort out the failed optimization attempts
    
    res_filtered=filter_dataframe(results,tau,success_criterion,n)
    
    ##### filter out only coordinate vector of the returne optimizer by algorithm
    
    res_filtered_again=res_filtered.loc[:,cols[0:(n+5)]]
    df=[]
    
    
    if success_criterion==0:
        if computation_method_x==0:
            for j in range(len(computational_budget)):
                comp_budge_j=computational_budget[j]
                data_j=res_filtered_again[res_filtered_again.comp_budget==comp_budge_j]
                data_j_coordinate=data_j.loc[:,cols_coordinate[0:(n+0)]]
                for i in range(len(data_j_coordinate)):
                    array_i=np.array(data_j_coordinate.iloc[i,:])
                    solver_array=info_object.solver
                    abs_diff=np.abs(np.subtract(solver_array,array_i))
                    sum_abs_diff_coordinates=np.sum(abs_diff)
                    mae=sum_abs_diff_coordinates/array_i.size
                    stats=np.hstack((comp_budge_j,mae))
                    df.append(stats)
                    
            dataframe=pd.DataFrame(df)
            dframe_1=dataframe.rename(columns={0:"comp_budget",1:"MAE_x"})
        
            return dframe_1
        
        elif computation_method_x==1:
            for j in range(len(computational_budget)):
                comp_budge_j=computational_budget[j]
                data_j=res_filtered_again[res_filtered_again.comp_budget==comp_budge_j]
                data_j_coordinate=data_j.loc[:,cols_coordinate[0:(n+0)]]
                for i in range(len(data_j_coordinate)):
                    array_i=np.array(data_j_coordinate.iloc[i,:])
                    solver_array=info_object.solver
                    diff=np.subtract(solver_array,array_i)
                    diff_square=np.square(diff)
                    sum_diff_square=np.sum(diff_square)
                    weighted_square=sum_diff_square/array_i.size
                    rmse=np.sqrt(weighted_square)
                    stats=np.hstack((comp_budge_j,rmse))
                    df.append(stats)
                    
            dataframe=pd.DataFrame(df)
            dframe_1=dataframe.rename(columns={0:"comp_budget",1:"RMSE_x"})
        
            return dframe_1
    
    elif success_criterion==1:
        select_cols=('comp_budget','success_crit_f')
        df_frame_2=res_filtered.loc[:,select_cols]
        dframe_1=df_frame_2.rename(columns={0:"comp_budget",1:"abs_dev_f"})
        
        return dframe_1
    
    
    
def mean_deviation_frame(dataframe_deviation,computational_budget):
    df=[]
    for i in range(len(computational_budget)):
        comp_budget_i=computational_budget[i]
        deviation_res_i=dataframe_deviation[dataframe_deviation.comp_budget==comp_budget_i]
        deviation_array_i=np.array(deviation_res_i.iloc[:,1])
        mean_deviation_i=np.mean(deviation_array_i)
        stats=np.hstack((comp_budget_i,mean_deviation_i))
        df.append(stats)
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"comp_budget",1:"mean_distance"})
    return dframe_1



def get_deviation_results_all(results_list,tau,success_criterion,computational_budget,info_object,cols_algo_names,n,computation_method_x=None):
    
    ### results_list: list containing results dataframes of algorithms to consider
    ### tau: desired tolerance of success
    ### success_criterion: 0->x_criterion/ 1-> f_val_criterion
    ### computational_budget: budgets considered
    ### info-object: contains information about the objective function
    ### cols_algo_names: list containing the algorithm names
    ### computation method x: 0-> computes MAE as deviation measure/ 1-> computes RMSE as deviation measure
    
    df=[]
    df.append(np.array(computational_budget))
    
    for i in range(len(results_list)):
        
        result_i=results_list[i]
        deviation_results_i=get_deviation_results_correct(result_i,tau,success_criterion,computational_budget,info_object,n,computation_method_x)
        mean_frame_i=mean_deviation_frame(deviation_results_i,computational_budget)
        df.append(mean_frame_i.iloc[:,1])
    
    dataframe_success=pd.DataFrame(df)
    dataframe_success_1=dataframe_success.transpose()
    dataframe_success_2= dataframe_success_1.rename(columns=cols_algo_names)
    
    return dataframe_success_2






def deviation_profile_general(deviation_results_frame,color_vector,marker_vector,algo_names):
    plt.figure(figsize=(20,10))
    plt.xscale('log')
    plt.yscale('log')
    #success_results_frame_1=success_results_frame.drop('comp_budget', 1)
    for i in range(len(deviation_results_frame.columns)-1):
        plt.plot(deviation_results_frame.comp_budget, deviation_results_frame.iloc[:,i+1], color=color_vector[i], linestyle='dashed',
                 linewidth = 2,marker=marker_vector[i], markerfacecolor=color_vector[i], markersize=8,label=algo_names[i])
    plt.ylim(1e-7,100)
    plt.xlim(0,102000)
    #plt.xscale("log")
 
    # naming the x axis
    plt.xlabel('Comp_budget')
    # naming the y axis
    plt.ylabel('Deviation')
 
    # giving a title to my graph
    plt.title('Deviation_Profile')
    plt.legend()

    # function to show the plot
    return plt.show()



def filter_dataframe_converged(results,tau,success_criterion,n):
    
    #### add column that indicates whether success criterion was met-> converged
    
    cols_dev=('f_evals','comp_budget','f_val_opt','success_crit_x','success_crit_f','6','7','8','9','10','11','12','13','14','15')
    res_filtered=results.loc[:,cols_dev[0:(n+5)]]
    
    ##### we want to select all FAILED implementations in order to plot the deviation
    ###### add true vector which indicated whether computational budget was exceeded
    res_filtered.comp_budget=res_filtered['comp_budget'].astype(float)
    res_filtered.f_evals=res_filtered['f_evals'].astype(float)
    
    vector_trues=np.less_equal(res_filtered.f_evals,res_filtered.comp_budget)
    
    
    res_filtered.loc[:,'trues_lesser_f_evals']=vector_trues
    def get_new_vector(x,y):
        df=[]
        for i in range(x.size):
            if x[i] & y[i]==True:
                df.append(True)
            else:
                df.append(False)
        array=np.array(df)
        return array    
    
    if success_criterion==0:
        
        #### initialise comparison
        trues_success_crit_x=np.less(results.success_crit_x,tau)
        trues_x=np.transpose(trues_success_crit_x)
        res_filtered.loc[:,'trues_success_crit_x']=trues_x
        conv=get_new_vector(res_filtered.trues_lesser_f_evals,res_filtered.trues_success_crit_x)
        res_filtered.loc[:,'converged']=conv
        
        
        return res_filtered
    
    if success_criterion==1:
        #### initialise comparison
        trues_success_crit_f=np.less(results.success_crit_f,tau)
        trues_f=np.transpose(trues_success_crit_f)
        res_filtered.loc[:,'trues_success_crit_f']=trues_f
        conv=get_new_vector(res_filtered.trues_lesser_f_evals,res_filtered.trues_success_crit_f)
        res_filtered.loc[:,'converged']=conv
        
        
        return res_filtered
    
    


def get_minimum_vector(results,tau,success_criterion,computational_budget,n):
    #### returns the vector of minimum FE-evals used by the algorithm to solve a problem for all Problems 
    ###(Problems: each starting point is a different problem)
    ### number of problems = number of different starting points considered
    
    no_starting_points=np.int(len(results)/len(computational_budget))
    no_budgets=np.int(len(computational_budget))
    converged_dframe=filter_dataframe_converged(results,tau,success_criterion,n)
    subset_frame=converged_dframe[['f_evals','converged']]
    f_evals_sub=np.array(subset_frame.f_evals)
    subset_converged=np.array(subset_frame.converged)
    def get_new_vector_f_eval(x,y):
        df=[]
        for i in range(x.size):
            if y[i]==True:
                df.append(x[i])
            else:
                df.append(np.infty)
        array=np.array(df)
        return array  
    
    settled=get_new_vector_f_eval(f_evals_sub,subset_converged)
    matrix_new_f_evals=np.matrix(settled).reshape(no_budgets,no_starting_points)
    transpose=matrix_new_f_evals.transpose()
    minimums=transpose.min(axis=1)
    return minimums


def get_minimum_vector_all_algos(results_list,tau,success_criterion,computational_budget,n):

        #### returns the number of minimum function evaluations needed to solve problem p of all algorithms so for problem p only the number 
        ### of FE's is displayed of the most efficient algorithm that solved the problem with the lowest number of FE's
        
        df=[]

        for i in range(len(results_list)):

            minimum_vector_i=get_minimum_vector(results_list[i],tau,success_criterion,computational_budget,n)
            df.append(minimum_vector_i.transpose())
        
        dataframe=pd.DataFrame(np.concatenate(df))
        matrix_r=np.matrix(dataframe).transpose()
        minimum_vector_all_algos=matrix_r.min(axis=1)
        minimum_vector_all_algos_1=np.array(minimum_vector_all_algos)
        
        return minimum_vector_all_algos_1
    
    
    
def get_matrix_ordered_single_algorithm(results,tau,success_criterion,computational_budget,n):
    no_starting_points=np.int(len(results)/len(computational_budget))
    no_budgets=np.int(len(computational_budget))
    converged_dframe=filter_dataframe_converged(results,tau,success_criterion,n)
    subset_frame=converged_dframe[['f_evals','converged']]
    f_evals_sub=np.array(subset_frame.f_evals)
    subset_converged=np.array(subset_frame.converged)
    def get_new_vector_f_eval(x,y):
        df=[]
        for i in range(x.size):
            if y[i]==True:
                df.append(x[i])
            else:
                df.append(np.infty)
        array=np.array(df)
        return array  
    
    settled=get_new_vector_f_eval(f_evals_sub,subset_converged)
    matrix_new_f_evals=np.matrix(settled).reshape(no_budgets,no_starting_points)
    matrix_transpose=matrix_new_f_evals.transpose()
    return matrix_transpose





def get_performance_metrics_one_problem(results_list,tau,success_criterion,computational_budget,algo_names,n):

    minimum_vector=get_minimum_vector_all_algos(results_list,tau,success_criterion,computational_budget,n)

    df=[]


    for i in range(len(results_list)):
        matrix_i=get_matrix_ordered_single_algorithm(results_list[i],tau,success_criterion,computational_budget,n)
        metrics=matrix_i/minimum_vector
        metrics_reshaped=metrics.reshape(1,len(results_list[i]))
        df.append(metrics_reshaped)
    
    data_frame=pd.DataFrame(np.concatenate(df))
    data_frame_transposed=data_frame.transpose()
    data_frame_1=data_frame_transposed.rename(columns=algo_names)
    return data_frame_1








########### general optimization function######################

def minimization_guvenen_all(f,computational_budget,algo,x_0,n,problem_info,x_tol_abs,f_tol_abs,local_tolerance_x=None,local_tolerance_f=None):
    
    
    
    
    ##################################     Input that need to be specified ################################
    #######################################################################################################
    
    
    ## x_tol: is the absolute Tolerance allowed
    ## f_tol: tolerance in function value allowed
    ## f: we need to specify the objective function
    ## computational budget: is a vector that contains different computational budgets between 0 and 10^5
    ## algo: specify the algorithm you want to use from the nlopt library -> argument has to have the form:
    ######## nlopt.algorithm_name e.g. nlopt.GN_ISRES for ISRES Algorithm
    ## algorithm: specify the algorithm 
    ## x_0: contains the randomly generated starting points -> pandas dataframe containing starting values
    ## n: number of dimensions the problem should have
    ## problem_info: object that that contains known information about the objective function 
                    ## as for example the domain
                    ## the best solver
                    ## function value of the best solver etc
    ### If you want to stop the optimization routine when the x_tol_abs for convergence is met 
    ########   -> plug in -inf for f_tol_abs
    ##### If you want to stop the optimization routine when the f_tol_abs convergence criterion is met specify:
    ######## -> x_tol_abs=-inf
    
    ######################################      Output       ################################################
    
    #### returns a dataframe containing:
    #### a vector of the optimizer -> columns #6-15 coordinate vector
    #### the function value of the optimizer -> next column                ##### this is done 100 times
    #### number of function evaluations -> next columns                      #### for all 100 starting points
    ### accuracy measures as specified in Guvenen et al. 
    
   ###### define the polishing search at the end of each global optimization
    fwrapped=lambda x,grad:f(x)

    polishing_optimizer=nlopt.opt(nlopt.LN_NELDERMEAD,n)
    polishing_optimizer.set_lower_bounds(problem_info.lower_bound)
    polishing_optimizer.set_upper_bounds(problem_info.upper_bound)
    polishing_optimizer.set_min_objective(fwrapped)
    polishing_optimizer.set_xtol_abs(-inf) ### in order to force that for every algorithm and optimization at the local stage are 1000 F-evals are made
    polishing_optimizer.set_ftol_abs(-inf) ### 
    polishing_optimizer.set_maxeval(1000)
    
    
    if algo==19:
        
        global_optimum=nlopt.opt(nlopt.GN_CRS2_LM,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    elif algo==42:
        
        global_optimum=nlopt.opt(nlopt.GN_ESCH,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    
    
    elif algo==35:
        
        global_optimum=nlopt.opt(nlopt.GN_ISRES,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    
    
    
    
    elif algo==20:
        
        global_optimum=nlopt.opt(nlopt.GN_MLSL,n)
        local_opt=nlopt.opt(nlopt.LN_NELDERMEAD,n)
        local_opt.set_lower_bounds(problem_info.lower_bound)
        local_opt.set_upper_bounds(problem_info.upper_bound)
        local_opt.set_min_objective(fwrapped)
        local_opt.set_xtol_abs(local_tolerance_x)
        local_opt.set_ftol_abs(local_tolerance_f)
        global_optimum.set_local_optimizer(local_opt)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
     
    elif algo==34:
        
        global_optimum=nlopt.opt(nlopt.GN_MLSL,n)
        local_opt=nlopt.opt(nlopt.LN_BOBYQA,n)
        local_opt.set_lower_bounds(problem_info.lower_bound)
        local_opt.set_upper_bounds(problem_info.upper_bound)
        local_opt.set_xtol_abs(local_tolerance_x)
        local_opt.set_ftol_abs(local_tolerance_f)
        local_opt.set_min_objective(fwrapped)
        global_optimum.set_local_optimizer(local_opt)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1

    elif algo==22:
        
        global_optimum=nlopt.opt(nlopt.GN_MLSL_LDS,n)
        local_opt=nlopt.opt(nlopt.LN_NELDERMEAD,n)
        local_opt.set_lower_bounds(problem_info.lower_bound)
        local_opt.set_upper_bounds(problem_info.upper_bound)
        local_opt.set_xtol_abs(local_tolerance_x)
        local_opt.set_ftol_abs(local_tolerance_f)
        local_opt.set_min_objective(fwrapped)
        global_optimum.set_local_optimizer(local_opt)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    




    
    
###################


def automate_result_generation(f,computational_budget,algo_list,x_0,n,problem_info,x_tol_abs,f_tol_abs,path,file_names,
                               local_tolerance_x_list,local_tolerance_f_list):
    
    for i in range(len(algo_list)):
        results_i=minimization_guvenen_all(f,computational_budget,algo_list[i],x_0,n,problem_info,x_tol_abs,
                                           f_tol_abs,local_tolerance_x_list[i],local_tolerance_f_list[i])
        results_i.to_csv(path+'/'+file_names[i]+'.csv')
    


###############


#### additional function that might be of interest later



def filter_dataframe_success(results,tau,success_criterion):
    
    #### this filter function selects all successful implementations
    
    cols_dev=('f_evals','comp_budget','f_val_opt','success_crit_x','success_crit_f','6','7','8','9','10','11','12','13','14','15')
    res_filtered=results.loc[:,cols_dev]
    
    ##### we want to select all FAILED implementations in order to plot the deviation
    ###### add true vector which indicated whether computational budget was exceeded
    res_filtered.comp_budget=res_filtered['comp_budget'].astype(float)
    res_filtered.f_evals=res_filtered['f_evals'].astype(float)
    
    vector_trues=np.less_equal(res_filtered.f_evals,res_filtered.comp_budget)
    
    
    res_filtered.loc[:,'trues_lesser_f_evals']=vector_trues
    
    if success_criterion==0:
        
        #### initialise comparison
        trues_success_crit_x=np.less(results.success_crit_x,tau)
        trues_x=np.transpose(trues_success_crit_x)
        res_filtered.loc[:,'trues_success_crit_x']=trues_x
        res_filtered_finished=res_filtered.drop(res_filtered[(res_filtered.trues_lesser_f_evals==False)
                                                             &(res_filtered.trues_success_crit_x==False)].index)
        res_filtered_finished_2=res_filtered_finished.drop(res_filtered[(res_filtered.trues_lesser_f_evals==True)
                                                                        &(res_filtered.trues_success_crit_x==False)].index)
        res_filtered_finished_3=res_filtered_finished_2.drop(res_filtered[(res_filtered.trues_lesser_f_evals==False)
                                                                          &(res_filtered.trues_success_crit_x==True)].index)
        return res_filtered_finished_3
    
    if success_criterion==1:
        #### initialise comparison
        trues_success_crit_f=np.less(results.success_crit_f,tau)
        trues_f=np.transpose(trues_success_crit_f)
        res_filtered.loc[:,'trues_success_crit_f']=trues_f
        res_filtered_finished=res_filtered.drop(res_filtered[(res_filtered.trues_lesser_f_evals==False)
                                                             &(res_filtered.trues_success_crit_f==False)].index)
        res_filtered_finished_2=res_filtered_finished.drop(res_filtered[(res_filtered.trues_lesser_f_evals==True)
                                                                        &(res_filtered.trues_success_crit_f==False)].index)
        res_filtered_finished_3=res_filtered_finished_2.drop(res_filtered[(res_filtered.trues_lesser_f_evals==False)
                                                                          &(res_filtered.trues_success_crit_f==True)].index)
        return res_filtered_finished_3


        
        


    

################## automate result generation two dimensional test functions




def minimization_guvenen_2D(f,computational_budget,algo,x_0,n,problem_info,x_tol_abs,f_tol_abs,local_tolerance_x=None,local_tolerance_f=None):
    
    
    
    
    ##################################     Input that need to be specified ################################
    #######################################################################################################
    
    
    ## x_tol: is the absolute Tolerance allowed
    ## f_tol: tolerance in function value allowed
    ## f: we need to specify the objective function
    ## computational budget: is a vector that contains different computational budgets between 0 and 10^5
    ## algo: specify the algorithm you want to use from the nlopt library -> argument has to have the form:
    ######## nlopt.algorithm_name e.g. nlopt.GN_ISRES for ISRES Algorithm
    ## algorithm: specify the algorithm 
    ## x_0: contains the randomly generated starting points -> pandas dataframe containing starting values
    ## n: number of dimensions the problem should have
    ## problem_info: object that that contains known information about the objective function 
                    ## as for example the domain
                    ## the best solver
                    ## function value of the best solver etc
    ### If you want to stop the optimization routine when the x_tol_abs for convergence is met 
    ########   -> plug in -inf for f_tol_abs
    ##### If you want to stop the optimization routine when the f_tol_abs convergence criterion is met specify:
    ######## -> x_tol_abs=-inf
    
    ######################################      Output       ################################################
    
    #### returns a dataframe containing:
    #### a vector of the optimizer -> columns #6-15 coordinate vector
    #### the function value of the optimizer -> next column                ##### this is done 100 times
    #### number of function evaluations -> next columns                      #### for all 100 starting points
    ### accuracy measures as specified in Guvenen et al. 
    
   ###### define the polishing search at the end of each global optimization
    #fwrapped=lambda x,grad:f(x)
    fwrapped=f

    polishing_optimizer=nlopt.opt(nlopt.LN_NELDERMEAD,n)
    polishing_optimizer.set_lower_bounds(problem_info.lower_bound)
    polishing_optimizer.set_upper_bounds(problem_info.upper_bound)
    polishing_optimizer.set_min_objective(fwrapped)
    polishing_optimizer.set_xtol_abs(-inf) ### in order to force that for every algorithm and optimization at the local stage are 1000 F-evals are made
    polishing_optimizer.set_ftol_abs(-inf) ### 
    polishing_optimizer.set_maxeval(1000)
    
    
    if algo==19:
        
        global_optimum=nlopt.opt(nlopt.GN_CRS2_LM,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer,grad)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    elif algo==42:
        
        global_optimum=nlopt.opt(nlopt.GN_ESCH,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer,grad)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    
    
    elif algo==35:
        
        global_optimum=nlopt.opt(nlopt.GN_ISRES,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer,grad)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    
    
    
    
    elif algo==20:
        
        global_optimum=nlopt.opt(nlopt.GN_MLSL,n)
        local_opt=nlopt.opt(nlopt.LN_NELDERMEAD,n)
        local_opt.set_lower_bounds(problem_info.lower_bound)
        local_opt.set_upper_bounds(problem_info.upper_bound)
        local_opt.set_min_objective(fwrapped)
        local_opt.set_xtol_abs(local_tolerance_x)
        local_opt.set_ftol_abs(local_tolerance_f)
        global_optimum.set_local_optimizer(local_opt)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer,grad)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
     
    elif algo==34:
        
        global_optimum=nlopt.opt(nlopt.GN_MLSL,n)
        local_opt=nlopt.opt(nlopt.LN_BOBYQA,n)
        local_opt.set_lower_bounds(problem_info.lower_bound)
        local_opt.set_upper_bounds(problem_info.upper_bound)
        local_opt.set_xtol_abs(local_tolerance_x)
        local_opt.set_ftol_abs(local_tolerance_f)
        local_opt.set_min_objective(fwrapped)
        global_optimum.set_local_optimizer(local_opt)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer=global_optimum.optimize(np.array(x_0.iloc[i]))
                #optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer,grad)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1

    elif algo==22:
        
        global_optimum=nlopt.opt(nlopt.GN_MLSL_LDS,n)
        local_opt=nlopt.opt(nlopt.LN_NELDERMEAD,n)
        local_opt.set_lower_bounds(problem_info.lower_bound)
        local_opt.set_upper_bounds(problem_info.upper_bound)
        local_opt.set_xtol_abs(local_tolerance_x)
        local_opt.set_ftol_abs(local_tolerance_f)
        local_opt.set_min_objective(fwrapped)
        global_optimum.set_local_optimizer(local_opt)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer_1=global_optimum.optimize(np.array(x_0.iloc[i]))
                optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=f(optimizer,grad)
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    


##########################


def automate_result_generation_2D(f,computational_budget,algo_list,x_0,n,problem_info,x_tol_abs,f_tol_abs,path,file_names,
                               local_tolerance_x_list,local_tolerance_f_list):
    
    for i in range(len(algo_list)):
        results_i=minimization_guvenen_2D(f,computational_budget,algo_list[i],x_0,n,problem_info,x_tol_abs,
                                           f_tol_abs,local_tolerance_x_list[i],local_tolerance_f_list[i])
        results_i.to_csv(path+'/'+file_names[i]+'.csv')
    
###################

    
#######################################

########### minimization function for the application

###################################################


def minimization_guvenen_application(f,computational_budget,algo,x_0,n,problem_info,x_tol_abs,f_tol_abs,constraint):
    
    
    
    
    
    ##################################     Input that need to be specified ################################
    #######################################################################################################
    
    
    ## x_tol: is the absolute Tolerance allowed
    ## f_tol: tolerance in function value allowed
    ## f: we need to specify the objective function
    ## computational budget: is a vector that contains different computational budgets between 0 and 10^5
    ## algo: specify the algorithm you want to use from the nlopt library -> argument has to have the form:
    ######## nlopt.algorithm_name e.g. nlopt.GN_ISRES for ISRES Algorithm
    ## algorithm: specify the algorithm 
    ## x_0: contains the randomly generated starting points -> pandas dataframe containing starting values
    ## n: number of dimensions the problem should have
    ## problem_info: object that that contains known information about the objective function 
                    ## as for example the domain
                    ## the best solver
                    ## function value of the best solver etc
    ### If you want to stop the optimization routine when the x_tol_abs for convergence is met 
    ########   -> plug in -inf for f_tol_abs
    ##### If you want to stop the optimization routine when the f_tol_abs convergence criterion is met specify:
    ######## -> x_tol_abs=-inf
    
    ######################################      Output       ################################################
    
    #### returns a dataframe containing:
    #### a vector of the optimizer -> columns #6-15 coordinate vector
    #### the function value of the optimizer -> next column                ##### this is done 100 times
    #### number of function evaluations -> next columns                      #### for all 100 starting points
    ### accuracy measures as specified in Guvenen et al. 
    
   ###### define the polishing search at the end of each global optimization
    #fwrapped=lambda x,grad:f(x)
    #constraint_wrapped= lambda x,grad:constraint(x)
    fwrapped=f
    constraint_wrapped=constraint

    
    if algo==25:
        
        global_optimum=nlopt.opt(nlopt.LN_COBYLA,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        global_optimum.add_equality_constraint(constraint_wrapped,1e-8)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer=global_optimum.optimize(np.array(x_0.iloc[i]))
                #optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=global_optimum.last_optimum_value()
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
    
    
    
    
    if algo==40:
        
        global_optimum=nlopt.opt(nlopt.LD_SLSQP,n)
        global_optimum.set_lower_bounds(problem_info.lower_bound)
        global_optimum.set_upper_bounds(problem_info.upper_bound)
        global_optimum.set_min_objective(fwrapped)
        global_optimum.set_xtol_abs(x_tol_abs)
        global_optimum.set_ftol_abs(f_tol_abs)
        global_optimum.add_equality_constraint(constraint_wrapped,1e-8)
        comp_budge=np.array(computational_budget)
        df=[]
        for j in range(comp_budge.size):
            comp_budge_j=comp_budge[j]
            global_optimum.set_maxeval(comp_budge_j)
            for i in range(len(x_0)):
                #start_point_i=np.array(x_0.iloc[i])
                optimizer=global_optimum.optimize(np.array(x_0.iloc[i]))
                #optimizer=polishing_optimizer.optimize(optimizer_1)
                function_val=global_optimum.last_optimum_value()
                num_evals=np.array(global_optimum.get_numevals())
            #### define accuracy measures
                comp_budget=comp_budge_j
                abs_diff=np.abs(np.subtract(problem_info.solver,optimizer)) ### account for multiple glob min missing
                success_crit_x=np.amax(abs_diff)
                success_crit_f=np.abs(np.subtract(problem_info.solver_function_value,function_val))
            
                information=np.hstack((problem_info.name,function_val,num_evals,comp_budget,success_crit_x,success_crit_f,optimizer))
                df.append(information)
            
        
        dframe=pd.DataFrame(df)
        dframe_1=dframe.rename(columns={0:"problem",1:"f_val_opt",2:"f_evals",
                                        3:"comp_budget",4:"success_crit_x",5:"success_crit_f"})
        
        return dframe_1
    
        
        
        
        
############## Data Pofile application

def data_profile_application(success_results_frame,color_vector,marker_vector,algo_names):
    plt.figure(figsize=(20,10))
    #plt.xscale('log')
    #success_results_frame_1=success_results_frame.drop('comp_budget', 1)
    for i in range(len(success_results_frame.columns)-1):
        plt.plot(success_results_frame.comp_budget, success_results_frame.iloc[:,i+1], color=color_vector[i], linestyle='dashed',
                 linewidth = 2,marker=marker_vector[i], markerfacecolor=color_vector[i], markersize=8,label=algo_names[i])
    plt.ylim(-0.05,1.1)
    plt.xlim(0,1500)
    #plt.xscale("log")
 
    # naming the x axis
    plt.xlabel('Comp_budget')
    # naming the y axis
    plt.ylabel('Success_Rate')
 
    # giving a title to my graph
    plt.title('Data_Profile')
    plt.legend()

    # function to show the plot
    return plt.show()



def deviation_profile_application(deviation_results_frame,color_vector,marker_vector,algo_names):
    plt.figure(figsize=(20,10))
    #plt.xscale('log')
    plt.yscale('log')
    #success_results_frame_1=success_results_frame.drop('comp_budget', 1)
    for i in range(len(deviation_results_frame.columns)-1):
        plt.plot(deviation_results_frame.comp_budget, deviation_results_frame.iloc[:,i+1], color=color_vector[i], linestyle='dashed',
                 linewidth = 2,marker=marker_vector[i], markerfacecolor=color_vector[i], markersize=8,label=algo_names[i])
    plt.ylim(1e-7,100)
    plt.xlim(0,102000)
    #plt.xscale("log")
 
    # naming the x axis
    plt.xlabel('Comp_budget')
    # naming the y axis
    plt.ylabel('Deviation')
 
    # giving a title to my graph
    plt.title('Deviation_Profile')
    plt.legend()

    # function to show the plot
    return plt.show()

        
####################################################################
#################################################################### Functions for performance profiles


def get_converged_vector_revised(results,tau,success_criterion,computational_budget,n):
    ### returns a vector that sets the number of FEs to infinity for all problems algorithm i did not solved 
    ###(Problems: each starting point is a different problem)
    ### number of problems = number of different starting points considered
    
    no_starting_points=np.int(len(results)/len(computational_budget))
    no_budgets=np.int(len(computational_budget))
    converged_dframe=filter_dataframe_converged(results,tau,success_criterion,n)
    subset_frame=converged_dframe[['f_evals','converged']]
    f_evals_sub=np.array(subset_frame.f_evals)
    subset_converged=np.array(subset_frame.converged)
    def get_new_vector_f_eval(x,y):
        df=[]
        for i in range(x.size):
            if y[i]==True:
                df.append(x[i])
            else:
                df.append(np.infty)
        array=np.array(df)
        return array  
    
    settled=get_new_vector_f_eval(f_evals_sub,subset_converged)
    #matrix_new_f_evals=np.matrix(settled).reshape(no_budgets,no_starting_points)
    #transpose=matrix_new_f_evals.transpose()
    #minimums=transpose.min(axis=1)
    return settled


def get_performance_metrics_revised(results_list,tau,success_criterion,computational_budget,cols,n):
    
    df=[]
    
    for i in range (len(results_list)):
        converged_vector_i=get_converged_vector_revised(results_list[i],tau,success_criterion,computational_budget,n)
        df.append(converged_vector_i)
        
    ##### transform df into matrix
    matrix_converged=np.matrix(df)
    matrix_converged_transpose=matrix_converged.transpose()
    minimum_vector=matrix_converged.min(axis=0)
    params=matrix_converged/minimum_vector
    params_transpose=params.transpose()
    
    dataframe=pd.DataFrame(np.concatenate(params_transpose))
    dataframe_1=dataframe.rename(columns=cols)
    return dataframe_1



def get_performance_metrics_alpha(performance_metrics_object,alpha,cols):
    
    rf=[]
    #rf.append(alpha)
    
    for i in range(len(performance_metrics_object.columns)):
        
        df=[]
        
        array_metrics_i=np.array(performance_metrics_object.iloc[:,i])
        
        for j in range(len(alpha)):
            
            alpha_j=alpha[j]
            less_vector=np.less(array_metrics_i,alpha_j)
            sum_less=sum(less_vector)
            df.append(sum_less)
        
        array_results_i=np.array(df)
        
        rf.append(array_results_i)
        
    dataframe=pd.DataFrame(rf)
    dataframe_1=dataframe.transpose()
    dataframe_2=dataframe_1.rename(columns=cols)
    
    return dataframe_2




def get_metrics_aggregated(metrics_list_object,no_start_points,no_budgets,cols,alpha):
    
    all_points_considered=no_start_points*no_budgets*len(metrics_list_object)
    
    list_1=[]
    
    ####create new list:
    
    for i in range(len(metrics_list_object)):
        
        matrix_i=np.matrix(metrics_list_object[i])
        list_1.append(matrix_i)
        
    
    
    
    
    def sum_of_list(l,n):
    
        if n == 0:
            return l[n]
        return l[n] + sum_of_list(l,n-1)
    
    sum_list=sum_of_list(list_1,len(metrics_list_object)-1)
    sum_list_prob=sum_list/all_points_considered
    dframe=pd.DataFrame(sum_list_prob)
    dframe_1=dframe.rename(columns=cols)
    dframe_1.loc[:,'alpha']=alpha
    
    
    return dframe_1





def performance_profile(metrics_agg_frame,color_vector,marker_vector,algo_names):
    plt.figure(figsize=(20,10))
    #plt.xscale('log')
    #success_results_frame_1=success_results_frame.drop('comp_budget', 1)
    for i in range(len(metrics_agg_frame.columns)-1):
        plt.plot(metrics_agg_frame.alpha, success_results_frame.iloc[:,i], color=color_vector[i], linestyle='dashed',
                 linewidth = 2,marker=marker_vector[i], markerfacecolor=color_vector[i], markersize=8,label=algo_names[i])
    plt.ylim(-0.05,1.1)
    plt.xlim(0,30)
    #plt.xscale("log")
 
    # naming the x axis
    plt.xlabel('alpha')
    # naming the y axis
    plt.ylabel('rho')
 
    # giving a title to my graph
    plt.title('Performance profile')
    plt.legend()

    # function to show the plot
    return plt.show()
    




