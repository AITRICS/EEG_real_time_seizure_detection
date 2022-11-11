import os
import numpy as np

from control.config import args


class experiment_results_validation:
    def __init__(self, args):
        self.args = args
        
        if args.task_type == "binary":
            self.auc_final_list = []
            self.apr_final_list = []
            self.f1_final_list = []
            self.tpr_final_list = []
            self.tnr_final_list = []
        
        elif args.task_type == "multiclassification":
            self.weighted_auc_final_list = []
            self.weighted_apr_final_list = []
            self.weighted_f1_final_list = []
            
            self.unweighted_auc_final_list = []
            self.unweighted_apr_final_list = []
            self.unweighted_f1_final_list = []
            
            self.all_auc_final_list = []
            self.all_apr_final_list = []
            self.all_f1_final_list = []

        else:
            exit(1)


    def results_all_seeds(self, list_of_test_results_per_seed):
        if args.task_type == "binary":
            os.system("echo  \'#######################################\'")
            os.system("echo  \'##### Final validation results per seed #####\'")
            os.system("echo  \'#######################################\'")
            seed, result, tpr, tnr = list_of_test_results_per_seed
            os.system("echo  \'seed_case:{} -- auc: {}, apr: {}, f1_score: {}, tpr: {}, tnr: {}\'".\
                format(seed, str(result[0]), str(result[1]), str(result[2]), str(tpr), str(tnr)))
                
            self.auc_final_list.append(result[0])
            self.apr_final_list.append(result[1])
            self.f1_final_list.append(result[2])
            self.tpr_final_list.append(tpr)
            self.tnr_final_list.append(tnr)
        
        elif args.task_type == "multiclassification":
            os.system("echo  \'#######################################\'")
            os.system("echo  \'##### Final validation results per seed #####\'")
            os.system("echo  \'#######################################\'")
            
            seed, result, result_aucs, result_aprs, result_f1scores, tprs, fnrs, tnrs, fprs, fdrs, ppvs = list_of_test_results_per_seed

            os.system("echo  \'multi_weighted: auc: {}, apr: {}, f1_score: {}\'".format(str(result[0]), str(result[2]), str(result[4])))
            os.system("echo  \'multi_unweighted: auc: {}, apr: {}, f1_score: {}\'".format(str(result[1]), str(result[3]), str(result[5])))
            os.system("echo  \'##### Each class Validation results #####\'")
            seizure_list = self.args.num_to_seizure_items
            results = []
            for idx, seizure in enumerate(seizure_list):
                results.append("Label:{} auc:{} apr:{} f1:{} tpr:{} fnr:{} tnr:{} fpr:{} fdr:{} ppv:{}".format(seizure,
                                                                                str(result_aucs[idx]), str(result_aprs[idx]), str(result_f1scores[idx]), 
                                                                                str(tprs[idx]), str(fnrs[idx]), str(tnrs[idx]), str(fprs[idx]), 
                                                                                str(fdrs[idx]), str(ppvs[idx])))

            for i in results:
                os.system("echo  \'{}\'".format(i))
                
            self.weighted_auc_final_list.append(result[0])
            self.weighted_apr_final_list.append(result[2])
            self.weighted_f1_final_list.append(result[4])
            
            self.unweighted_auc_final_list.append(result[1])
            self.unweighted_apr_final_list.append(result[3])
            self.unweighted_f1_final_list.append(result[5])
            
            self.all_auc_final_list.append(result_aucs)
            self.all_apr_final_list.append(result_aprs)
            self.all_f1_final_list.append(result_f1scores)
        
        else:
            exit(1)


    def results_per_cross_fold(self):
        print("##########################################################################################")
        if args.task_type == "binary":
            os.system("echo  \'{} fold cross validation results each fold with\'".format(str(self.args.nfold)))
            os.system("echo  \'Total mean average -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.mean(self.auc_final_list)), 
                str(np.mean(self.apr_final_list)), 
                str(np.mean(self.f1_final_list)), 
                str(np.mean(self.tpr_final_list)), 
                str(np.mean(self.tnr_final_list))))

            os.system("echo  \'Total mean std -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.std(self.auc_final_list)), 
                str(np.std(self.apr_final_list)), 
                str(np.std(self.f1_final_list)), 
                str(np.std(self.tpr_final_list)), 
                str(np.std(self.tnr_final_list))))
            
        elif args.task_type == "multiclassification":
            os.system("echo  \'{} fold cross validation results each fold with\'".format(str(self.args.nfold)))
            os.system("echo  \'Total mean average weighted -- auc: {}, apr: {}, f1_score: {}\'".format(
                str(np.mean(self.weighted_auc_final_list)), 
                str(np.mean(self.weighted_apr_final_list)), 
                str(np.mean(self.weighted_f1_final_list))))

            os.system("echo  \'Total mean std weighted -- auc: {}, apr: {}, f1_score: {}\'".format(
                str(np.std(self.weighted_auc_final_list)), 
                str(np.std(self.weighted_apr_final_list)), 
                str(np.std(self.weighted_f1_final_list))))
        
            os.system("echo  \'Total mean average unweighted -- auc: {}, apr: {}, f1_score: {}\'".format(
                str(np.mean(self.unweighted_auc_final_list)), 
                str(np.mean(self.unweighted_apr_final_list)), 
                str(np.mean(self.unweighted_f1_final_list))))

            os.system("echo  \'Total mean std unweighted -- auc: {}, apr: {}, f1_score: {}\'".format(
                str(np.std(self.unweighted_auc_final_list)), 
                str(np.std(self.unweighted_apr_final_list)), 
                str(np.std(self.unweighted_f1_final_list))))
            
            all_auc_final_list_np = np.array(self.all_auc_final_list)
            all_apr_final_list_np = np.array(self.all_apr_final_list)
            all_f1_final_list_np = np.array(self.all_f1_final_list)
            
            seizure_list = self.args.num_to_seizure_items
            for indx, seizure in enumerate(seizure_list):
                msg = f"echo  \' "
                
                msg_mean = msg + f"means || auc: {str(np.round(np.mean(all_auc_final_list_np[:,indx]), 4))}, " +\
                                 f"apr: {str(np.round(np.mean(all_apr_final_list_np[:,indx]), 4))}, " +\
                                 f"f1_score: {str(np.round(np.mean(all_f1_final_list_np[:,indx]), 4))}\'"
                
                msg_std = msg + f"stds || auc: {str(np.round(np.std(all_auc_final_list_np[:,indx]), 4))}, " +\
                                 f"apr: {str(np.round(np.std(all_apr_final_list_np[:,indx]), 4))}, " +\
                                 f"f1_score: {str(np.round(np.std(all_f1_final_list_np[:,indx]), 4))}\'"

                os.system(msg_mean)
                os.system(msg_std)
        
        else:
            exit(1)
            
class experiment_results:
    def __init__(self, args):
        self.args = args
        
        if args.task_type == "binary":
            self.auc_final_list = []
            self.apr_final_list = []
            self.f1_final_list = []
            self.tpr_final_list = []
            self.tnr_final_list = []
        
        elif args.task_type == "multiclassification":
            self.weighted_auc_final_list = []
            self.weighted_apr_final_list = []
            self.weighted_f1_final_list = []
            
            self.unweighted_auc_final_list = []
            self.unweighted_apr_final_list = []
            self.unweighted_f1_final_list = []
            
            self.all_auc_final_list = []
            self.all_apr_final_list = []
            self.all_f1_final_list = []

        else:
            exit(1)


    def results_all_seeds(self, list_of_test_results_per_seed):
        if args.task_type == "binary":
            os.system("echo  \'#######################################\'")
            os.system("echo  \'##### Final test results per seed #####\'")
            os.system("echo  \'#######################################\'")
            seed, result, tpr, tnr = list_of_test_results_per_seed
            os.system("echo  \'seed_case:{} -- auc: {}, apr: {}, f1_score: {}, tpr: {}, tnr: {}\'".\
                format(seed, str(result[0]), str(result[1]), str(result[2]), str(tpr), str(tnr)))
                
            self.auc_final_list.append(result[0])
            self.apr_final_list.append(result[1])
            self.f1_final_list.append(result[2])
            self.tpr_final_list.append(tpr)
            self.tnr_final_list.append(tnr)
        
        elif args.task_type == "multiclassification":
            os.system("echo  \'#######################################\'")
            os.system("echo  \'##### Final test results per seed #####\'")
            os.system("echo  \'#######################################\'")
            
            seed, result, result_aucs, result_aprs, result_f1scores, tprs, fnrs, tnrs, fprs, fdrs, ppvs = list_of_test_results_per_seed

            os.system("echo  \'multi_weighted: auc: {}, apr: {}, f1_score: {}\'".format(str(result[0]), str(result[2]), str(result[4])))
            os.system("echo  \'multi_unweighted: auc: {}, apr: {}, f1_score: {}\'".format(str(result[1]), str(result[3]), str(result[5])))
            os.system("echo  \'##### Each class test results #####\'")
            seizure_list = self.args.num_to_seizure_items
            results = []
            for idx, seizure in enumerate(seizure_list):
                results.append("Label:{} auc:{} apr:{} f1:{} tpr:{} fnr:{} tnr:{} fpr:{} fdr:{} ppv:{}".format(seizure,
                                                                                str(result_aucs[idx]), str(result_aprs[idx]), str(result_f1scores[idx]), 
                                                                                str(tprs[idx]), str(fnrs[idx]), str(tnrs[idx]), str(fprs[idx]), 
                                                                                str(fdrs[idx]), str(ppvs[idx])))

            for i in results:
                os.system("echo  \'{}\'".format(i))
                
            self.weighted_auc_final_list.append(result[0])
            self.weighted_apr_final_list.append(result[2])
            self.weighted_f1_final_list.append(result[4])
            
            self.unweighted_auc_final_list.append(result[1])
            self.unweighted_apr_final_list.append(result[3])
            self.unweighted_f1_final_list.append(result[5])
            
            self.all_auc_final_list.append(result_aucs)
            self.all_apr_final_list.append(result_aprs)
            self.all_f1_final_list.append(result_f1scores)
        
        else:
            exit(1)


    def results_per_cross_fold(self):
        print("##########################################################################################")
        if args.task_type == "binary":
            os.system("echo  \'{} fold cross test results each fold with\'".format(str(self.args.nfold)))
            os.system("echo  \'Total mean average -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.mean(self.auc_final_list)), 
                str(np.mean(self.apr_final_list)), 
                str(np.mean(self.f1_final_list)), 
                str(np.mean(self.tpr_final_list)), 
                str(np.mean(self.tnr_final_list))))

            os.system("echo  \'Total mean std -- auc: {}, apr: {}, f1_score: {}, tnr: {}, tpr: {}\'".format(
                str(np.std(self.auc_final_list)), 
                str(np.std(self.apr_final_list)), 
                str(np.std(self.f1_final_list)), 
                str(np.std(self.tpr_final_list)), 
                str(np.std(self.tnr_final_list))))
            
        elif args.task_type == "multiclassification":
            os.system("echo  \'{} fold cross test results each fold with\'".format(str(self.args.nfold)))
            os.system("echo  \'Total mean average weighted -- auc: {}, apr: {}, f1_score: {}\'".format(
                str(np.mean(self.weighted_auc_final_list)), 
                str(np.mean(self.weighted_apr_final_list)), 
                str(np.mean(self.weighted_f1_final_list))))

            os.system("echo  \'Total mean std weighted -- auc: {}, apr: {}, f1_score: {}\'".format(
                str(np.std(self.weighted_auc_final_list)), 
                str(np.std(self.weighted_apr_final_list)), 
                str(np.std(self.weighted_f1_final_list))))
        
            os.system("echo  \'Total mean average unweighted -- auc: {}, apr: {}, f1_score: {}\'".format(
                str(np.mean(self.unweighted_auc_final_list)), 
                str(np.mean(self.unweighted_apr_final_list)), 
                str(np.mean(self.unweighted_f1_final_list))))

            os.system("echo  \'Total mean std unweighted -- auc: {}, apr: {}, f1_score: {}\'".format(
                str(np.std(self.unweighted_auc_final_list)), 
                str(np.std(self.unweighted_apr_final_list)), 
                str(np.std(self.unweighted_f1_final_list))))
            
            all_auc_final_list_np = np.array(self.all_auc_final_list)
            all_apr_final_list_np = np.array(self.all_apr_final_list)
            all_f1_final_list_np = np.array(self.all_f1_final_list)
            
            seizure_list = self.args.num_to_seizure_items
            for indx, seizure in enumerate(seizure_list):
                msg = f"echo  \' "
                
                msg_mean = msg + f"means || auc: {str(np.round(np.mean(all_auc_final_list_np[:,indx]), 4))}, " +\
                                 f"apr: {str(np.round(np.mean(all_apr_final_list_np[:,indx]), 4))}, " +\
                                 f"f1_score: {str(np.round(np.mean(all_f1_final_list_np[:,indx]), 4))}\'"
                
                msg_std = msg + f"stds || auc: {str(np.round(np.std(all_auc_final_list_np[:,indx]), 4))}, " +\
                                 f"apr: {str(np.round(np.std(all_apr_final_list_np[:,indx]), 4))}, " +\
                                 f"f1_score: {str(np.round(np.std(all_f1_final_list_np[:,indx]), 4))}\'"

                os.system(msg_mean)
                os.system(msg_std)
        
        else:
            exit(1)
            
            
            
            