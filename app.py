import streamlit as st
import os
import tempfile
import time
from AFPILD_Predict import predict_locations, plot_footstep_locations
import matplotlib.pyplot as plt

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 1
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'tmp_file_path' not in st.session_state:
    st.session_state.tmp_file_path = None

def reset_app():
    st.session_state.stage = 1
    st.session_state.predictions = None
    if st.session_state.tmp_file_path and os.path.exists(st.session_state.tmp_file_path):
        os.unlink(st.session_state.tmp_file_path)
    st.session_state.tmp_file_path = None

# Page 1: File Upload
def show_upload_page():
    st.title("Audio Footstep Localization")
    st.write("Upload an audio file to predict footstep locations")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
    
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.tmp_file_path = tmp_file.name
        
        st.success("File uploaded successfully!")
        if st.button("Process Audio"):
            st.session_state.stage = 2
            st.rerun()

# Page 2: Processing
def show_processing_page():
    st.title("Processing Audio")
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate processing steps
    steps = [
        "Loading audio file...",
        "Detecting footsteps...",
        "Extracting features...",
        "Making predictions...",
        "Generating visualization..."
    ]
    
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) / len(steps))
        time.sleep(1) 
    
    try:
        # Make predictions
        predictions = predict_locations("best_model.h5", st.session_state.tmp_file_path)
        st.session_state.predictions = predictions
        
        # Move to results page
        st.session_state.stage = 3
        st.rerun()
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        if st.button("Try Again"):
            reset_app()
            st.rerun()

# Page 3: Results
def show_results_page():
    st.title("Footstep Localization Results")
    
    if st.session_state.predictions:
        # Add CSS to style the container
        st.markdown("""
            <style>
                .results-container {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                }
                .stTable {
                    background-color: white !important;
                    border-radius: 5px !important;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
                }
            </style>
        """, unsafe_allow_html=True)

        # Add metrics at the top
        metrics_cols = st.columns(4)
        total_steps = len(st.session_state.predictions)
        total_distance = sum(
            ((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)**0.5 
            for p1, p2 in zip(st.session_state.predictions[:-1], st.session_state.predictions[1:])
        )
        duration = st.session_state.predictions[-1]['time'] - st.session_state.predictions[0]['time']
        avg_speed = total_distance / duration if duration > 0 else 0

        # Calculate most common subject
        subjects = [p['subject'] for p in st.session_state.predictions]
        most_common_subject = max(set(subjects), key=subjects.count)
        subject_confidence = (subjects.count(most_common_subject) / len(subjects)) * 100

        # Display subject identification prominently
        st.markdown(f"""
            <div style='padding: 20px; background-color: #e6f3ff; border-radius: 10px; margin-bottom: 20px; text-align: center;'>
                <h2 style='margin: 0;'>Subject Identified: Person {most_common_subject}</h2>
                <p style='margin: 5px 0 0 0;'>Confidence: {subject_confidence:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        with metrics_cols[0]:
            st.metric("Total Steps", f"{total_steps}")
        with metrics_cols[1]:
            st.metric("Total Distance", f"{total_distance:.2f} m")
        with metrics_cols[2]:
            st.metric("Duration", f"{duration:.2f} s")
        with metrics_cols[3]:
            st.metric("Average Speed", f"{avg_speed:.2f} m/s")

        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Visualization", "📋 Detailed Data"])
        
        with tab1:
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            # Add visualization controls
            viz_cols = st.columns([3, 1])
            with viz_cols[1]:
                st.markdown("### Visualization Controls")
                show_arrows = st.checkbox("Show Movement Arrows", value=True)
                show_colorbar = st.checkbox("Show Time Colorbar", value=True)
                marker_size = st.slider("Marker Size", 50, 200, 100)
                
            with viz_cols[0]:
                # Plot the results with custom settings
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
                
                # Extract coordinates
                x_coords = [p['x'] for p in st.session_state.predictions]
                y_coords = [p['y'] for p in st.session_state.predictions]
                times = [p['time'] for p in st.session_state.predictions]
                
                # Create scatter plot
                scatter = ax.scatter(x_coords, y_coords, 
                                   c=times if show_colorbar else None,
                                   cmap='viridis',
                                   s=marker_size, 
                                   alpha=0.6)
                
                if show_colorbar:
                    plt.colorbar(scatter, label='Time (seconds)')
                
                if show_arrows:
                    for i in range(len(x_coords)-1):
                        ax.arrow(x_coords[i], y_coords[i],
                                x_coords[i+1]-x_coords[i], y_coords[i+1]-y_coords[i],
                                color='gray', alpha=0.3, 
                                head_width=0.05, length_includes_head=True)
                
                ax.set_xlabel('X Coordinate (m)')
                ax.set_ylabel('Y Coordinate (m)')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab2:
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            # Add search and filter options
            search_col, filter_col = st.columns([2, 1])
            with search_col:
                search = st.text_input("🔍 Search in data")
            with filter_col:
                sort_by = st.selectbox("Sort by", ["Time (s)", "X (m)", "Y (m)"])
            
            # Prepare and filter data
            data = [{
                'Time (s)': f"{p['time']:.2f}",
                'X (m)': f"{p['x']:.2f}",
                'Y (m)': f"{p['y']:.2f}",
                'Step #': i+1
            } for i, p in enumerate(st.session_state.predictions)]
            
            # Apply search filter
            if search:
                data = [d for d in data if any(search.lower() in str(v).lower() for v in d.values())]
            
            # Apply sorting
            if sort_by:
                data = sorted(data, key=lambda x: float(x[sort_by]) if sort_by != "Subject" else x[sort_by])
            
            # Display the table with styling
            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Step #": st.column_config.NumberColumn(
                        "Step #",
                        help="Step sequence number",
                        format="%d"
                    ),
                    "Time (s)": st.column_config.NumberColumn(
                        "Time (s)",
                        help="Time of footstep detection",
                        format="%.2f"
                    ),
                    "X (m)": st.column_config.NumberColumn(
                        "X (m)",
                        help="X coordinate in meters",
                        format="%.2f"
                    ),
                    "Y (m)": st.column_config.NumberColumn(
                        "Y (m)",
                        help="Y coordinate in meters",
                        format="%.2f"
                    )
                }
            )
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Add export options
    if st.session_state.predictions:
        st.markdown("### Export Options")
        export_cols = st.columns(3)
        with export_cols[0]:
            if st.button("Export to CSV"):
                # Add CSV export functionality here
                st.download_button(
                    label="Download CSV",
                    data="\n".join([
                        "Time (s),X (m),Y (m)",
                        *[f"{p['time']:.2f},{p['x']:.2f},{p['y']:.2f}"
                          for p in st.session_state.predictions]
                    ]),
                    file_name="footstep_predictions.csv",
                    mime="text/csv"
                )
    
    # Add button to process another file
    st.sidebar.button("Process Another File", on_click=reset_app)

# Main app logic
def main():
    st.set_page_config(page_title="Audio Footstep Localization", layout="wide")
    
    if st.session_state.stage == 1:
        show_upload_page()
    elif st.session_state.stage == 2:
        show_processing_page()
    elif st.session_state.stage == 3:
        show_results_page()

if __name__ == "__main__":
    main() 