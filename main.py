import os
import numpy as np
import pandas as pd
import plotly.express as px

class WindTurbinePipeline:
    def __init__(self, input_path, output_dir, plots_dir):
        """Initializes the file pathways."""
        self.input_path = input_path
        self.output_dir = output_dir
        self.plots_dir = plots_dir
        self.df = None
        
    def ingest_data(self):
        """Step 1: Load the raw SCADA CSV file safely."""
        try:
            print("⏳ Step 1: Ingesting raw SCADA data...")
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"Missing raw data file at: {self.input_path}")
            
            self.df = pd.read_csv(self.input_path)
            print(f"✅ Ingestion successful. Loaded {len(self.df)} rows.")
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            
    def clean_data(self):
        """Step 2: Clean data and isolate the unique 8-12 m/s wind speed slice."""
        try:
            print("⏳ Step 2: Cleaning data and applying unique operational filter...")
            
            self.df['Date/Time'] = pd.to_datetime(self.df['Date/Time'], format='%d %m %Y %H:%M')
            self.df = self.df[(self.df['Wind Speed (m/s)'] >= 8) & (self.df['Wind Speed (m/s)'] <= 12)].copy()
            
            self.df.dropna(inplace=True)
            self.df.drop_duplicates(inplace=True)
            
            os.makedirs(self.output_dir, exist_ok=True)
            cleaned_file_path = os.path.join(self.output_dir, 'dataset_cleaned.csv')
            self.df.to_csv(cleaned_file_path, index=False)
            
            print(f"✅ Cleaning complete. Filtered dataset contains {len(self.df)} rows.")
        except Exception as e:
            print(f"❌ Cleaning failed: {e}")

    def analyze_metrics(self, column_name):
        """Step 3: Compute engineering metrics using pure NumPy arrays."""
        try:
            print(f"⏳ Step 3: Calculating engineering metrics for column: [{column_name}]...")
            data_array = self.df[column_name].to_numpy()
            
            metrics = {
                "Mean": np.mean(data_array),
                "Median": np.median(data_array),
                "Std Dev": np.std(data_array),
                "Variance": np.var(data_array)
            }
            
            print(f"📊 --- {column_name} STATS ---")
            print(f"   • Mean Output: {metrics['Mean']:.2f} kW")
            print(f"   • Median Output: {metrics['Median']:.2f} kW")
            print(f"   • Std Deviation: {metrics['Std Dev']:.2f}")
            return metrics
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return None

    def generate_plots(self):
        """Step 4: Generate 3 static engineering plots and 2 animated charts."""
        try:
            print("⏳ Step 4: Generating static and animated engineering charts...")
            os.makedirs(self.plots_dir, exist_ok=True)
            
            # Sort data chronologically
            sorted_df = self.df.sort_values(by='Date/Time').copy()
            
            # Take a subset so the animations compile efficiently without crashing
            plot_df = sorted_df.head(500).copy()
            
            # Create a string version of the time for the animation timeline
            plot_df['Time_Label'] = plot_df['Date/Time'].dt.strftime('%Y-%m-%d %H:%M')
            
            # ==========================================
            # 📊 STATIC PLOTS (3 REQUIRED)
            # ==========================================
            
            # --- Static Plot 1: Power Curve Scatter ---
            fig1 = px.scatter(
                plot_df, 
                x='Wind Speed (m/s)', 
                y='LV ActivePower (kW)',
                title='Static 1: Wind Speed vs Active Power (8-12 m/s Filter)',
                labels={'Wind Speed (m/s)': 'Wind Speed (m/s)', 'LV ActivePower (kW)': 'Power Output (kW)'},
                template='plotly_dark'
            )
            fig1.write_image(os.path.join(self.plots_dir, 'static_plot1.png'))
            print("💾 Saved: outputs/static_plot1.png")
            
            # --- Static Plot 2: Time Series Timeline ---
            fig2 = px.line(
                plot_df, 
                x='Date/Time', 
                y='LV ActivePower (kW)',
                title='Static 2: Active Power Production Timeline',
                template='plotly_dark'
            )
            fig2.write_image(os.path.join(self.plots_dir, 'static_plot2.png'))
            print("💾 Saved: outputs/static_plot2.png")
            
            # --- Static Plot 3: Power Output Distribution ---
            fig3 = px.histogram(
                plot_df,
                x='LV ActivePower (kW)',
                nbins=30,
                title='Static 3: Frequency Distribution of Active Power Output',
                template='plotly_dark',
                color_discrete_sequence=['#00CC96']
            )
            fig3.write_image(os.path.join(self.plots_dir, 'static_plot3.png'))
            print("💾 Saved: outputs/static_plot3.png")
            
            # ==========================================
            # 🎬 ANIMATED PLOTS (2 REQUIRED)
            # ==========================================
            
            # --- Animated Plot 1: Power Output Evolution ---
            print("🎬 Compiling Animation 1...")
            fig_anim1 = px.scatter(
                plot_df,
                x='Wind Speed (m/s)',
                y='LV ActivePower (kW)',
                animation_frame='Time_Label',  
                range_x=[7.5, 12.5],            
                range_y=[-100, 4000],              
                color='Wind Direction (°)',     
                title='Animation 1: Power Output Evolution over Time',
                template='plotly_dark'
            )
            fig_anim1.write_html(os.path.join(self.plots_dir, 'animation1.html'))
            print("💾 Saved: outputs/animation1.html")
            
            # --- Animated Plot 2: Wind Speed vs Direction Tracking ---
            print("🎬 Compiling Animation 2...")
            fig_anim2 = px.scatter(
                plot_df,
                x='Wind Speed (m/s)',
                y='Wind Direction (°)',
                animation_frame='Time_Label',
                range_x=[7.5, 12.5],
                range_y=[-10, 370],
                color='LV ActivePower (kW)',
                title='Animation 2: Wind Speed vs Direction Shifts over Time',
                template='plotly_dark'
            )
            fig_anim2.write_html(os.path.join(self.plots_dir, 'animation2.html'))
            print("💾 Saved: outputs/animation2.html")
            
            print("✅ All 3 static and 2 animated files successfully updated!")
        except Exception as e:
            print(f"❌ Visualization failed: {e}")

# The execution block that runs the whole pipeline top-to-bottom
if __name__ == "__main__":
    pipeline = WindTurbinePipeline(
        input_path="data/dataset_original.csv", 
        output_dir="data",
        plots_dir="outputs"
    )
    
    pipeline.ingest_data()
    pipeline.clean_data()
    pipeline.analyze_metrics(column_name="LV ActivePower (kW)")
    pipeline.generate_plots()
