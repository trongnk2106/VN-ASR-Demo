# Vietnamese Speech to Text Web Application

## Overview

This web application is designed for Vietnamese Speech to Text conversion, allowing users to transcribe audio files or speak directly into the microphone. The application is built using [Streamlit](https://streamlit.io/) and utilizes the powerful Wav2Vec 2.0 model for speech recognition. The model has been fine-tuned from the pre-trained [nguyenvulebinh/wav2vec2-base-vietnamese-250h](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) on a dataset obtained from the [BKAI2023 competition](#acknowledgments).

## Features

- **Audio Input Options:**
  - Import audio files in `.wav` or `.mp3` format.
  - Directly speak into the microphone for real-time transcription.

- **Wav2Vec 2.0 Model:**
  - The speech recognition is powered by the Wav2Vec 2.0 model, known for its accuracy in transcribing spoken language.

- **Fine-Tuned Model:**
  - The model has been fine-tuned specifically for the Vietnamese language using the pre-trained [nguyenvulebinh/wav2vec2-base-vietnamese-250h](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h).

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Install required packages: `pip install -r requirements.txt`

### Running the Application

1. Clone the repository: `git clone https://github.com/trongnk2106/VN-ASR-Demo.git`
2. Navigate to the project directory: `cd VN-ASR-Demo`
3. Run the application: `streamlit run app.py`
4. Access the web application at [http://localhost:8501](http://localhost:8501) in your web browser.

## Usage

1. **File Input:**
   - Click on the "Upload Audio File" button and select a `.wav` or `.mp3` file for transcription.

2. **Microphone Input:**
   - Click on the "Start Microphone" button and speak into your microphone for real-time transcription.

3. **View Transcription:**
   - The transcribed text will be displayed on the web page.

## Acknowledgments

- The Wav2Vec 2.0 model used in this project is based on the [nguyenvulebinh/wav2vec2-base-vietnamese-250h](https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h) pre-trained model.
- Training data is sourced from the [BKAI2023 competition](#acknowledgments).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Issues and Contributions

If you encounter any issues or would like to contribute to the project, please open an issue or pull request on the [GitHub repository](https://github.com/your-repository).

