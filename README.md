# Youtube Comments Summarizer

This is a Python script that extracts the comments of a specific YouTube video and summarizes them into positive, negative, doubt, request, and statement categories. It uses data preprocessing, zero-shot classification with the facebook/bart-large-mnli model, and the Google API.

# Flow Chart
![Flow Chart](https://raw.githubusercontent.com/sjarunvenkat/YouTube-Comment-Summarizer/main/Flowchart.jpg)

# Screenshots
![App Screenshot](https://raw.githubusercontent.com/sjarunvenkat/YouTube-Comment-Summarizer/main/welcomepage.png)

![App Screenshot](https://raw.githubusercontent.com/sjarunvenkat/YouTube-Comment-Summarizer/main/outputpage.png)


## Requirements
- Python 3.x
- Django
- Google API key
- transformers package
- google-auth package
- google-auth-oauthlib package
- google-auth-httplib2 package
- pandas package
- nltk package

## Usage

1. Clone this repository.
2. Install the required packages
3. Replace the DEVELOPER_KEY in the script with your own Google API key.
4. In the same script, replace the videoId parameter with the ID of the YouTube video whose comments you want to extract.
5. Run the script using the following command in your terminal or command prompt:
```python
python manage.py runserver
```
6. After running the script, the extracted comments will be saved in a CSV file named comments.csv.
7. The summarized comments will be saved in the following CSV files:
   - positive.csv: Comments classified as positive and further classified as doubt, request, or statement.
   - negative.csv: Comments classified as negative and further classified as doubt, request, or statement.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
