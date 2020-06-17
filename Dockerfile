FROM python:3
ADD app.py /
EXPOSE 8503
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD streamlit run webapp/app.py