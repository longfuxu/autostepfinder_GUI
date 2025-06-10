import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

from autostepfinder import AutoStepFinder, StepFinderParameters

def plot_fit(data, fit, params):
    """Generates the main interactive plot of data vs. fit using Plotly."""
    fig = go.Figure()

    time_axis = np.arange(len(data)) * params.resolution

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=data,
        mode='lines',
        name='Data',
        line=dict(color='cornflowerblue', width=1.5),
        opacity=0.9
    ))

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=fit,
        mode='lines',
        name='Fit',
        line=dict(color='darkorange', width=2)
    ))

    fig.update_layout(
        title="Stepfinder Fit",
        xaxis_title=f"Time (s)",
        yaxis_title="Position (A.U.)",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        hovermode="x unified"
    )
    return fig

def plot_s_curves(s_curves, n_found, params):
    """Generates the interactive S-curve evaluation plot using Plotly."""
    fig = go.Figure()

    step_number = np.arange(1, len(s_curves[:, 0]))
    s_curve_r1 = s_curves[1:, 0]
    s_curve_r2 = s_curves[1:, 1]

    fig.add_trace(go.Scatter(x=step_number, y=s_curve_r1, mode='lines', name='Round 1', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=step_number, y=s_curve_r2, mode='lines', name='Round 2', line=dict(color='blue')))

    fig.add_hline(y=params.s_max_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")

    stepno_r1 = n_found[0]
    if stepno_r1 > 0 and stepno_r1 < len(s_curve_r1) :
        fig.add_trace(go.Scatter(x=[stepno_r1], y=[s_curve_r1[stepno_r1-1]], mode='markers', name='Optimal (R1)', marker=dict(color='black', size=10)))

    stepno_r2 = n_found[1]
    if stepno_r2 > 0 and stepno_r2 < len(s_curve_r2):
        fig.add_trace(go.Scatter(x=[stepno_r2], y=[s_curve_r2[stepno_r2-1]], mode='markers', name='Optimal (R2)', marker=dict(color='blue', size=10)))
    
    fig.update_layout(
        title="Multi-pass S-curves",
        xaxis_title="Iteration Number",
        yaxis_title="S-Value",
        hovermode="x unified"
    )
    fig.update_yaxes(rangemode="tozero")
    return fig

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="AutoStepFinder")

st.title("🤖 AutoStepFinder")
st.markdown("A Python implementation of the automated step detection method for single-molecule analysis.")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    
    uploaded_file = st.file_uploader("Upload Data File", type=['txt', 'csv', 'dat'])
    
    st.markdown("---")
    st.subheader("Analysis Parameters")

    s_max_threshold = st.number_input(
        "S-Max Threshold", 
        min_value=0.0, max_value=1.0, value=0.1, step=0.01,
        help="The primary tuning parameter. Threshold for accepting a round of fitting based on the S-curve peak. Higher values find fewer, more significant steps. Lower values find more steps, including smaller or noisier ones."
    )

    fit_range = st.number_input(
        "Max Iterations", 
        min_value=10, max_value=100000, value=10000, step=100,
        help="Maximum number of step-finding iterations in the first pass. Limits analysis time on very long traces. The default is usually sufficient."
    )

    resolution = st.number_input(
        "Time Resolution (s/point)",
        min_value=0.0, value=1.0, step=0.001, format="%.5f",
        help="The time duration of a single data point. This scales the x-axis of the plots to show time in seconds."
    )

    with st.expander("Advanced Settings"):
        st.markdown("**Fitting Method**")
        fit_mode = st.radio(
            "Plateau Level Calculation", 
            ['mean', 'median'], index=0, horizontal=True,
            help="Method to determine the level of each plateau. 'Mean' is standard, but 'median' is more robust to outliers within a plateau."
        )

        st.markdown("**Step Rejection**")
        local_step_merge = st.toggle(
            "Enable Local Step Merging", value=True,
            help="A cleaning step for noisy data. If enabled, second-round steps with high relative error are rejected and merged back into larger plateaus."
        )
        error_tolerance = st.slider(
            "Error Tolerance", 
            min_value=0.5, max_value=5.0, value=2.0, step=0.1,
            disabled=not local_step_merge,
            help="Controls the stringency of the Local Step Merging. Higher tolerance keeps more second-pass steps; lower tolerance rejects more."
        )
        
        st.markdown("**Cosmetic**")
        overshoot = st.slider(
            "Overshoot", 
            min_value=0.5, max_value=2.0, value=1.0, step=0.05,
            help="Fine-tuning parameter. Multiplies the number of steps picked from the S-curve. >1 forces more steps, <1 forces fewer."
        )

# --- Main Page ---
if uploaded_file is not None:
    try:
        # To convert to a string based IO object
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # Read data, assuming single column or time in first and data in second
        raw_data = np.loadtxt(stringio)
        if raw_data.ndim > 1 and raw_data.shape[1] > 1:
            st.info("Multi-column data detected. Using the second column for analysis.")
            data = raw_data[:, 1]
        else:
            data = raw_data.flatten()
            
        st.success(f"Successfully loaded '{uploaded_file.name}' with {len(data)} data points.")

        # Instantiate parameters from GUI
        params = StepFinderParameters(
            s_max_threshold=s_max_threshold,
            overshoot=overshoot,
            fit_range=fit_range,
            resolution=resolution,
            fit_mode=fit_mode,
            local_step_merge=local_step_merge,
            error_tolerance=error_tolerance
        )

        # Run analysis
        with st.spinner('Analyzing data... This may take a moment.'):
            finder = AutoStepFinder(params=params)
            final_fit, final_steps, s_curves, n_found_steps = finder.run(data)
        
        st.header("Analysis Results")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Fit Plot", "📈 S-Curve", "📋 Step Properties", "📊 Distributions"])

        with tab1:
            st.plotly_chart(plot_fit(data, final_fit, params), use_container_width=True)

        with tab2:
            st.plotly_chart(plot_s_curves(s_curves, n_found_steps, params), use_container_width=True)

        with tab3:
            st.metric("Total Steps Found", len(final_steps))
            st.dataframe(final_steps, use_container_width=True)
            
            # Add download buttons
            st.download_button(
                label="Download Fits (CSV)",
                data=pd.DataFrame({'data': data, 'fit': final_fit}).to_csv(index=False).encode('utf-8'),
                file_name=f"{uploaded_file.name.split('.')[0]}_fits.csv",
                mime='text/csv',
            )
            st.download_button(
                label="Download Step Properties (CSV)",
                data=final_steps.to_csv(index=False).encode('utf-8'),
                file_name=f"{uploaded_file.name.split('.')[0]}_properties.csv",
                mime='text/csv',
            )

        with tab4:
            if not final_steps.empty:
                st.subheader("Step Property Distributions")
                col1, col2 = st.columns(2)

                with col1:
                    fig_size = px.histogram(
                        final_steps, 
                        x="step_size", 
                        nbins=30,
                        title="Step Size Distribution"
                    )
                    st.plotly_chart(fig_size, use_container_width=True)

                with col2:
                    log_y_dwell = st.checkbox("Log scale for Dwell Time Y-axis")
                    fig_dwell = px.histogram(
                        final_steps, 
                        x="dwell_time_after", 
                        nbins=30,
                        title="Dwell Time Distribution (After Step)"
                    )
                    if log_y_dwell:
                        fig_dwell.update_yaxes(type="log")
                    st.plotly_chart(fig_dwell, use_container_width=True)
            else:
                st.info("No steps found to plot distributions.")

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        st.exception(e)

else:
    st.info("Upload a data file in the sidebar to begin analysis.") 