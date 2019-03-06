# GalNest Usage
1. run_galnest.sh to run MultiNest on simulated visibility dataset.
2. run_compute_mode_SNR.sh on the pickled output from MultiNest first to return a dataframe with SNRs for each mode computed. 
3. Run get_results.py and specify the pickle output from 1. and a SNR threshold as inputs. Modes in the output dataframe from this will be in the final sample of modes if 'final' == 1 for their row.
