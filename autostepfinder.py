#!/usr/bin/env python3
"""
AutoStepFinder - A tool for automated step detection in time series data

This application provides an interface for detecting step changes in trace data
with options for both sudden steps and gradual changes.
"""

import os
import sys
import tkinter as tk
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class StepFinderParameters:
    """
    A data class to hold the parameters for the step-finding algorithm.

    Attributes:
        s_max_threshold (float): Threshold for accepting a fit round based on the S-curve peak.
            A higher value makes the criterion for accepting a round of step-fitting more stringent,
            leading to fewer steps. A lower value is less stringent and may find more, potentially
            noisy, steps. Typical values are 0.05-0.2. Defaults to 0.1.
        overshoot (float): Multiplier for the number of steps picked from the S-curve.
            Values > 1.0 can force the algorithm to include more steps than deemed optimal,
            while values < 1.0 will include fewer. Can be used for fine-tuning. Defaults to 1.0.
        fit_range (int): Maximum number of steps to iterate through in the first pass.
            Limits the computational cost for very long traces. Should be smaller than
            len(data)/4. Defaults to 10000.
        resolution (float): The time resolution of the data (e.g., seconds per data point).
            Used for plotting. Defaults to 1.0.
        fit_mode (str): Method to determine the level of each plateau ('mean' or 'median').
            'median' is more robust to outliers within a plateau. Defaults to 'mean'.
        min_window_size (int): Minimum number of data points required for a plateau to be
            considered for splitting. Defaults to 3.
        local_step_merge (bool): If enabled, second-round steps with high relative error will be
            rejected, merging them back into larger steps. This helps clean up noise. Defaults to True.
        error_tolerance (float): Tolerance for outlier detection when setting the relative error
            threshold for step merging. Only active if local_step_merge is True. Higher values
            are more permissive, keeping more steps. Defaults to 2.0.
    """
    s_max_threshold: float = 0.1
    overshoot: float = 1.0
    fit_range: int = 10000
    resolution: float = 1.0
    fit_mode: str = 'mean'  # 'mean' or 'median'
    min_window_size: int = 3
    local_step_merge: bool = True
    error_tolerance: float = 2.0


class AutoStepFinder:
    """
    A Python implementation of the Loeff-Kerssemakers AutoStepfinder algorithm.
    """
    def __init__(self, params: StepFinderParameters = StepFinderParameters()):
        self.params = params

    def run(self, data: np.ndarray) -> tuple:
        """
        The main entry point for running the step-finding algorithm on a dataset.
        This corresponds to 'autostepfinder_mainloop' in the MATLAB script.

        Args:
            data (np.ndarray): The 1D time series data.

        Returns:
            tuple: (final_fit, final_steps, s_curves, n_found_steps_per_round)
                   - final_fit: The final fitted step trace.
                   - final_steps: A DataFrame with the properties of the final steps.
                   - s_curves: The S-curves from both rounds.
                   - n_found_steps_per_round: Number of steps found in each round.
        """
        n_points = len(data)
        if n_points == 0:
            return np.array([]), pd.DataFrame(), np.zeros((10000, 2)), np.zeros(2, dtype=int)

        # Initialize variables
        residual = data.copy()
        fit = np.zeros_like(data)
        step_number_first_run = min(int(np.ceil(n_points / 4)), self.params.fit_range)
        s_curves = np.zeros((step_number_first_run + 1, 2))
        n_found_steps_per_round = np.zeros(2, dtype=int)
        full_split_log = pd.DataFrame(columns=['index', 'round'])

        # Core dual-pass loop
        for fit_round in range(1, 3):
            self.params.fit_range = step_number_first_run
            fit_residu, _, s_curve, split_indices, best_shot = self.stepfinder_core(residual)

            s_curves[:, fit_round - 1] = s_curve

            step_round_accept = (s_curve.max() > self.params.s_max_threshold)

            if step_round_accept:
                n_found_steps_per_round[fit_round - 1] = best_shot
                full_split_log = self._expand_split_log(full_split_log, split_indices, fit_round, best_shot)

            residual -= fit_residu
            fit += fit_residu
        
        if n_found_steps_per_round.max() == 0:
            print("No steps found.")
            # Return a flat trace if no steps are found
            final_fit = np.full_like(data, np.mean(data))
            return final_fit, pd.DataFrame(), s_curves, n_found_steps_per_round
        
        final_steps, final_fit = self._build_final_fit(data, full_split_log)
        
        return final_fit, final_steps, s_curves, n_found_steps_per_round

    def _build_final_fit(self, data: np.ndarray, split_log: pd.DataFrame) -> tuple:
        """
        Builds the final step fit from all retained indices, including
        error-based rejection of second-round steps.
        Corresponds to 'BuildFinalfit'.
        """
        best_shot = len(split_log[split_log['round'] < 3])
        steps_to_pick = int(round(self.params.overshoot * best_shot))
        
        best_list = split_log.head(steps_to_pick)
        candidate_loc = np.sort(best_list['index'].values)
        
        candidate_fit = self.get_fit_from_indices(data, candidate_loc)
        candidate_steps = self.get_step_table_from_fit(candidate_fit)
        
        candidate_steps = self._add_step_errors(data, candidate_steps)
        
        # Relative step error: error / |step_size|
        candidate_steps['rel_error'] = np.abs(candidate_steps['error'] / candidate_steps['step_size'])
        
        # Match round number back to steps
        step_to_round_map = best_list.set_index('index')['round']
        candidate_steps['round'] = candidate_steps['index'].map(step_to_round_map)

        if self.params.local_step_merge:
            # Keep round 1 steps and 'good' round 2 steps
            is_round1 = candidate_steps['round'] == 1
            rel_error_r1 = candidate_steps.loc[is_round1, 'rel_error'].dropna()
            
            if len(rel_error_r1) > 0:
                _, _, final_error_threshold = self._outlier_flag(rel_error_r1.values, self.params.error_tolerance)
                
                sel_merge = (candidate_steps['rel_error'] < 2 * final_error_threshold) | is_round1
                final_indices = candidate_steps.loc[sel_merge, 'index'].values
            else: # No round 1 steps, keep all candidates
                 final_indices = candidate_steps['index'].values
        else:
            final_indices = candidate_steps['index'].values
            
        final_fit = self.get_fit_from_indices(data, final_indices)
        final_steps = self.get_step_table_from_fit(final_fit)
        final_steps = self._add_step_errors(data, final_steps)
        
        # Add round number to final steps table
        final_steps['round'] = final_steps['index'].map(step_to_round_map)

        return final_steps, final_fit

    def _add_step_errors(self, data: np.ndarray, steps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the error for each step based on the standard deviation
        of the adjacent plateaus. Corresponds to 'AddStep_Errors'.
        """
        if steps_df.empty:
            steps_df['error'] = []
            return steps_df
            
        errors = []
        step_indices = steps_df['index'].values
        boundaries = np.concatenate(([-1], step_indices, [len(data) - 1]))

        for i in range(len(step_indices)):
            i1 = boundaries[i] + 1
            i2 = boundaries[i+1] + 1
            i3 = boundaries[i+2] + 1
            
            plateau_before = data[i1:i2]
            plateau_after = data[i2:i3]
            
            n_before = len(plateau_before)
            n_after = len(plateau_after)
            
            if n_before > 1 and n_after > 1:
                rms_before = np.std(plateau_before, ddof=1)
                rms_after = np.std(plateau_after, ddof=1)
                # Propagated error for the difference of two means
                error = np.sqrt(rms_before**2 / n_before + rms_after**2 / n_after)
            else:
                error = np.nan
            errors.append(error)
        
        steps_df['error'] = errors
        return steps_df

    @staticmethod
    def _outlier_flag(data: np.ndarray, tolerance: float, sig_change_threshold: float = 0.8) -> tuple:
        """
        Iteratively removes outliers from a dataset until the standard deviation stabilizes.
        Corresponds to 'Outlier_flag'.
        """
        if len(data) == 0:
            return np.array([]), np.array([]), np.nan
        
        flag = np.ones_like(data, dtype=bool)
        sigma = np.inf
        ratio = 0.0

        while ratio < sig_change_threshold:
            sigma_old = sigma
            
            selected_data = data[flag]
            if len(selected_data) < 2:
                break
                
            av = np.nanmedian(selected_data)
            sigma = np.nanstd(selected_data)
            
            if sigma_old == 0 or sigma == 0: # Avoid division by zero
                break
            
            ratio = sigma / sigma_old if sigma_old != np.inf else 0.0
            
            threshold = tolerance * sigma + av
            flag = (data - av) < threshold
        
        clean_data = data[flag]
        final_threshold = tolerance * np.nanstd(clean_data) + np.nanmedian(clean_data)
        
        return flag, clean_data, final_threshold
        
    @staticmethod
    def _expand_split_log(full_split_log: pd.DataFrame, split_indices: np.ndarray, 
                          fit_round: int, best_shot: int) -> pd.DataFrame:
        """
        Merges new split indices into the main log, avoiding duplicates.
        Corresponds to 'expand_split_log'.
        """
        new_log = pd.DataFrame({
            'index': split_indices,
            'round': 3  # Default round is 3 (duplicate/not chosen)
        })
        new_log.loc[:best_shot-1, 'round'] = fit_round
        
        if fit_round == 1:
            return pd.concat([full_split_log, new_log], ignore_index=True)
        else:
            if not full_split_log.empty:
                already_found = full_split_log[full_split_log['round'] < 3]['index']
                new_found = new_log[~new_log['index'].isin(already_found)]
                
                # Keep original good steps plus newly found steps
                return pd.concat([full_split_log[full_split_log['round'] < 3], new_found], ignore_index=True)
            else:
                return new_log

    def stepfinder_core(self, x: np.ndarray):
        """
        Performs a single-pass step-finding routine on the data.
        This corresponds to 'StepfinderCore' in the MATLAB script.
        """
        step_number = min(int(np.ceil(len(x) / 4)), self.params.fit_range)

        fit_x, _, s_raw, split_log = self._split_until_ready(x, step_number)

        s_peak_idx, s_final = self.eval_scurve(s_raw)
        
        # -1 because s_peak_idx is 1-based
        best_shot = int(round(min(s_peak_idx - 1, np.ceil(len(x) / 4))))

        # Sort the best 'n' indices
        index_list = sorted(split_log[:best_shot])
        
        final_fit = self.get_fit_from_indices(x, index_list)
        # The full step table is not needed here, only the fit on the residual.
        return final_fit, None, s_final, split_log, best_shot

    def _split_until_ready(self, x: np.ndarray, step_number: int):
        """
        Iteratively splits the data to find steps, generating an S-curve.
        Corresponds to 'Split_until_ready' in the MATLAB script.
        """
        n = len(x)
        fit_x = np.full(n, np.mean(x))
        s_curve = np.ones(step_number + 1)
        split_log = np.zeros(step_number, dtype=int)

        # 1. Initialize the plateau properties table 'f'
        # [istart, istop, split_idx, avg_left, avg_right, rank]
        istart, istop = 0, n - 1
        inxt, avl, avr, rankit, _ = self.splitfast(x)
        
        # Sentinel rows at top and bottom for easier indexing
        f_table = np.array([
            [-1, -1, -1, 0, 0, 0],  # Sentinel
            [istart, istop, inxt + istart - 1, avl, avr, rankit],
            [n, n, n, 0, 0, 0]    # Sentinel
        ], dtype=float)

        # 2. Build the first counter-fit
        c_fit_x = np.zeros_like(fit_x)
        i1, i2, i3 = 0, int(f_table[1, 2]), n - 1
        c_fit_x[i1:i2+1] = avl
        c_fit_x[i2+1:i3+1] = avr

        # 3. Iteratively split plateaus
        for c in range(1, step_number + 1):
            # Find the best plateau to split (max rankit)
            # Exclude sentinels and plateaus with zero rank
            valid_plateaus = (f_table[1:-1, 5] > 0) & \
                             ((f_table[1:-1, 1] - f_table[1:-1, 0]) > self.params.min_window_size)
            
            if not np.any(valid_plateaus):
                s_curve[c:] = s_curve[c-1] # No more valid splits, fill S-curve
                break

            f_valid_indices = np.where(valid_plateaus)[0] + 1
            best_f_idx = f_valid_indices[np.argmax(f_table[f_valid_indices, 5])]

            split_log[c-1] = int(f_table[best_f_idx, 2])
            
            fit_x = self._adapt_fit(f_table, best_f_idx, fit_x)
            f_table = self._expand_f_table(f_table, best_f_idx, x)
            c_fit_x = self._adapt_cfit(f_table, best_f_idx, c_fit_x, x)
            
            # Calculate S-value directly, avoiding complex qm/aqm updates
            s_curve[c] = np.mean((x - c_fit_x)**2) / np.mean((x - fit_x)**2)

        return fit_x, f_table, s_curve, split_log
    
    def _adapt_fit(self, f_table: np.ndarray, idx: int, fit_x: np.ndarray) -> np.ndarray:
        """Updates the fit trace with a new split."""
        i1 = int(f_table[idx, 0])       # start of plateau
        i2 = int(f_table[idx, 2])       # split point
        av1 = f_table[idx, 3]           # avg of left part
        
        i3 = int(f_table[idx, 2]) + 1   # start of right part
        i4 = int(f_table[idx, 1])       # end of plateau
        av2 = f_table[idx, 4]           # avg of right part

        fit_x[i1:i2+1] = av1
        fit_x[i3:i4+1] = av2
        return fit_x

    def _adapt_cfit(self, f_table: np.ndarray, idx: int, c_fit_x: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Updates the 'counter-fit' trace locally around the new split."""
        i1 = int(f_table[idx - 1, 2])
        i2 = int(f_table[idx, 2])
        i3 = int(f_table[idx + 1, 2])
        
        # Update the three local segments of the counter-fit
        if i1 < i2:
            c_fit_x[i1+1:i2+1] = np.mean(x[i1+1:i2+1])
        if i2 < i3:
            c_fit_x[i2+1:i3+1] = np.mean(x[i2+1:i3+1])

        return c_fit_x
        
    def _expand_f_table(self, f_table: np.ndarray, idx: int, x: np.ndarray) -> np.ndarray:
        """
        Replaces one row in the 'f' table with two new rows corresponding
        to the two new sub-plateaus.
        """
        # Left sub-plateau
        istart1 = int(f_table[idx, 0])
        istop1 = int(f_table[idx, 2])
        segment1 = x[istart1:istop1+1]
        inxt1, avl1, avr1, rankit1, _ = self.splitfast(segment1)
        n1 = [istart1, istop1, inxt1 + istart1 - 1, avl1, avr1, rankit1]

        # Right sub-plateau
        istart2 = int(f_table[idx, 2]) + 1
        istop2 = int(f_table[idx, 1])
        segment2 = x[istart2:istop2+1]
        inxt2, avl2, avr2, rankit2, _ = self.splitfast(segment2)
        n2 = [istart2, istop2, inxt2 + istart2 - 1, avl2, avr2, rankit2]

        f_table[idx, :] = n1
        f_table = np.insert(f_table, idx + 1, n2, axis=0)
        
        return f_table

    @staticmethod
    def splitfast(segment: np.ndarray) -> tuple:
        """
        Analyzes a 1D array 'segment' to find the best single-step fit.
        This is a Pythonic, vectorized implementation of 'Splitfast'.

        Args:
            segment (np.ndarray): The data segment to analyze.

        Returns:
            tuple: (index, level_before, level_after, rank, error_curve)
                   - index: The 1-based index of the split point (length of left plateau).
                   - level_before: Mean of the left plateau.
                   - level_after: Mean of the right plateau.
                   - rank: A metric for the quality of the split.
                   - error_curve: The chi-squared error curve.
        """
        w = len(segment)
        if w < 4:  # Minimal size for a split creating two plateaus of size >= 2
            return 0, np.mean(segment), np.mean(segment), 0, np.zeros(w-1)

        # Vectorized calculation of variance reduction
        t = np.arange(2, w - 1)  # Potential split points (ensuring plateaus of size >= 2)
        
        cumsum_segment = np.cumsum(segment)
        sum_segment = cumsum_segment[-1]
        av_all = sum_segment / w
        
        mean_l = cumsum_segment[t - 1] / t
        mean_r = (sum_segment - cumsum_segment[t - 1]) / (w - t)
        
        # Variance reduction term
        delta_chisq = (mean_l - av_all)**2 * t + (mean_r - av_all)**2 * (w - t)
        
        if len(delta_chisq) == 0:
            return 0, av_all, av_all, 0, np.zeros(w-1)

        best_idx_in_delta = np.argmax(delta_chisq)
        idx = t[best_idx_in_delta]  # Length of the left plateau for the best split
        
        avl = np.mean(segment[:idx])
        avr = np.mean(segment[idx:])
        
        rankit = (avr - avl)**2 * w
        
        errorcurve = np.zeros(w - 1)
        errorcurve[t - 1] = -delta_chisq / (w - 1)

        return idx, avl, avr, rankit, errorcurve

    def get_fit_from_indices(self, x: np.ndarray, index_list: list) -> np.ndarray:
        """
        Builds a step-fit curve from a list of step indices.
        Corresponds to 'Get_FitFromStepsindices'.

        Args:
            x (np.ndarray): The original data trace.
            index_list (list): 0-based indices of the last point of each plateau.

        Returns:
            np.ndarray: The fitted trace.
        """
        fit_x = np.zeros_like(x, dtype=float)
        boundaries = np.concatenate(([-1], index_list, [len(x) - 1]))

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i] + 1
            end_idx = boundaries[i+1] + 1
            segment = x[start_idx:end_idx]
            
            if len(segment) > 0:
                if self.params.fit_mode == 'mean':
                    level = np.nanmean(segment)
                elif self.params.fit_mode == 'median':
                    level = np.nanmedian(segment)
                else:
                    raise ValueError("fit_mode must be 'mean' or 'median'")
                fit_x[start_idx:end_idx] = level
                
        return fit_x

    @staticmethod
    def eval_scurve(s_raw: np.ndarray) -> tuple:
        """
        Evaluates the S-curve to find the optimal number of steps.
        Corresponds to 'Eval_Scurve'.

        Args:
            s_raw (np.ndarray): The raw S-curve values.

        Returns:
            tuple: (best_shot, s_final)
                   - best_shot: The optimal number of steps (1-based).
                   - s_final: The processed S-curve.
        """
        s_raw[s_raw < 1] = 1
        s2 = s_raw - 1
        baseline = np.linspace(0, s2[-1], len(s2))
        s3 = s2 - baseline
        best_shot = np.argmax(s3) + 1  # +1 for 1-based index
        s_final = s3
        return best_shot, s_final

    @staticmethod
    def get_step_table_from_fit(fit_x: np.ndarray) -> pd.DataFrame:
        """
        Extracts step properties from a fitted trace into a DataFrame.
        Corresponds to 'Get_StepTableFromFit'.

        Args:
            fit_x (np.ndarray): The fitted data trace.

        Returns:
            pd.DataFrame: A DataFrame containing properties of each detected step.
        """
        lx = len(fit_x)
        if lx < 2:
            return pd.DataFrame()
            
        t = np.arange(lx)
        diff_x = np.diff(fit_x)
        step_indices, = np.where(diff_x != 0)

        if len(step_indices) == 0:
            return pd.DataFrame()

        # Dwell times of the plateaus
        all_boundaries = np.concatenate(([-1], step_indices, [lx - 1]))
        dwell_times = np.diff(all_boundaries)

        steps_df = pd.DataFrame({
            'index': step_indices,  # 0-based index of point before step
            'time': t[step_indices],
            'level_before': fit_x[step_indices],
            'level_after': fit_x[step_indices + 1],
            'step_size': diff_x[step_indices],
            'dwell_time_before': dwell_times[:-1],
            'dwell_time_after': dwell_times[1:]
        })

        return steps_df

def main():
    """
    Main entry point for the AutoStepFinder application
    """
    try:
        # Get the directory containing this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Add the script directory to the Python path
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        # Change the working directory to the script directory
        os.chdir(script_dir)
        print(f"Working directory set to: {os.getcwd()}")
        
        # Import modules after setting up the path
        from gui import StepfinderGUI
        
        # Create and run the application
        print("Starting AutoStepFinder application")
        root = tk.Tk()
        app = StepfinderGUI(root)
        print("Entering main loop")
        root.mainloop()
        print("Application closed")
        
    except Exception as e:
        # Print the error for debugging
        import traceback
        traceback.print_exc()
        
        # Show error to user and wait for input before closing
        print(f"\nError starting application: {str(e)}")
        input("Press Enter to close...")
        sys.exit(1)

if __name__ == "__main__":
    main() 