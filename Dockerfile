# Immagine base: quella costruita col Dockerfile di PROST
#   docker build -t prost-base -f Dockerfile.prost .
FROM prost-base

WORKDIR /app

# Dipendenze Python del progetto (layer separato per sfruttare la cache)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Codice del progetto (ricordati: git submodule update --init prima del build,
# così ProcessPilot finisce nell'immagine)
COPY . .

# Sostituisce i path assoluti di Windows con quelli del container
RUN grep -rlI "C:/Users/alegi/Desktop/intoRDDL" /app --include="*.py" \
    | xargs -r sed -i 's#C:/Users/alegi/Desktop/intoRDDL#/app#g'

# Sovrascrive l'ENTRYPOINT di PROST (prost.sh).
# Con ENTRYPOINT in forma exec, gli argomenti di 'docker run' vengono
# passati a main.py: docker run intordd --dataset sepsis_preprocessed
ENTRYPOINT ["python", "main.py"]