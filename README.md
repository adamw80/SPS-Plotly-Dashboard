
# SPS Plotly Dashboard

A Plotly Dash web application designed to visualize and analyze synthetic patient data for a mental health clinic. This project includes a range of financial, no-show, and patient churn analyses and is deployed [here](https://portfolio-projects.fly.dev). Please note that the site can be slow to load due to the amount of data compiled.

**Note**: The data in this project is synthetic and not real patient data, ensuring compliance with HIPAA regulations. However, it is based on actual patient demographic statistics to maintain relevance.

## Project Overview

This dashboard provides insights into various aspects of a clinic’s operations, with sections dedicated to:
- **Financial Overview**: Revenue and profit breakdowns by quarter and financial forecasting.
- **No-Show Analysis**: Rates of appointment no-shows by age group, diagnosis, and improvement metrics.
- **Patient Churn Analysis**: Visualizations of patient churn trends, including diagnosis distribution, distance from the clinic, and other metrics affecting churn.

### Data Sources

The data used in this dashboard originates from the following projects:
- **[Appointment No-Show Analysis Pipeline](https://github.com/adamw80/SPS-Appointment-No-Show-Analysis-Pipeline)**: This project provided the core dataset for no-show analysis, offering insights into the factors affecting patient attendance.
- **[K-Means Patient Churn Analysis](https://github.com/adamw80/SPS-K-Means-Patient-Churn-Analysis)**: This project supplied the data for analyzing patient churn rates, focusing on trends in patient retention and engagement.

All data is synthetic but structured based on real patient demographics to ensure realistic and actionable insights.

## Project Structure

- **app.py**: Main application file containing the Plotly Dash code for layout, figures, and graph configurations.
- **Dockerfile**: Configuration for containerizing the application for deployment.
- **requirements.txt**: List of dependencies required to run the application.
- **notebook.ipynb**: Jupyter Notebook containing code for data preparation and initial graph generation.
- **assets/**: Directory with the logo and any additional static resources.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- **Python 3.8+**
- **Docker** (for containerized deployment)

### Running the App Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adamw80/SPS-Plotly-Dashboard.git
   cd SPS-Plotly-Dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   python app.py
   ```
   Access the dashboard at `http://127.0.0.1:8080`.

### Running with Docker

1. **Build the Docker image**:
   ```bash
   docker build -t sps-plotly-dashboard .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 8080:8080 sps-plotly-dashboard
   ```
   Open `http://127.0.0.1:8080` in your browser to view the dashboard.

## Deployment

This project is deployed on Fly.io. For deployment steps, refer to the Fly.io documentation or adjust the Dockerfile and `fly.toml` file as needed for your environment.

## License

This project is licensed under the MIT License.

## Acknowledgments

This project is built with synthetic data to protect patient privacy and comply with HIPAA guidelines. The data originates from analyses on appointment no-shows and patient churn trends, combining insights from multiple projects to ensure a comprehensive and informative dashboard.

## Contact
For questions or feedback, feel free to reach out:
- **Email:** Adam.RivardWalter@gmail.com  
- **GitHub:** [adamw80](https://github.com/adamw80)

