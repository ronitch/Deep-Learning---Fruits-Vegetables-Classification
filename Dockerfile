FROM python
LABEL maintainer="https://github.com/ronitch" 
LABEL version="1.0" 
LABEL description="docker image for fruits/vegetables classification ml model"
RUN pip install numpy pandas matplotlib plotly tensorflow streamlit keras IPython PIL pickle sklearn
EXPOSE 8501 
