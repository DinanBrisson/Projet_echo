from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pytz
from io import BytesIO
from PIL import Image
import base64

app = Flask(__name__)
app.secret_key = 'votre_cle_secrete'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Modèle pour les utilisateurs
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    histories = db.relationship('History', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# Modèle pour stocker l'historique des segmentations
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model = db.Column(db.String(50))
    result = db.Column(db.Text)
    surface_segmentee = db.Column(db.Float)
    autre_mesure = db.Column(db.Float)
    date = db.Column(db.DateTime, default=datetime.utcnow)

# Route pour la racine qui redirige vers la page de connexion
@app.route('/')
def index():
    return redirect(url_for('login'))

# Route d'inscription pour créer un compte utilisateur
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')

        # Vérifier la correspondance des mots de passe
        if password != confirm:
            flash('Les mots de passe ne correspondent pas.', 'error')
            return render_template('register.html')

        # Vérifier si l'utilisateur ou l'email existe déjà
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Nom d\'utilisateur ou email déjà utilisé.', 'error')
            return render_template('register.html')

        # Création du nouvel utilisateur
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Compte créé avec succès. Vous pouvez maintenant vous connecter.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


# Route de connexion
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['username'] = user.username
            session['user_id'] = user.id  # Stocker l'identifiant de l'utilisateur
            return redirect(url_for('menu'))
        else:
            flash("Nom d'utilisateur ou mot de passe incorrect.", 'error')
    return render_template('login.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route de sélection du modèle de segmentation et lancement du traitement
@app.route('/menu', methods=['GET', 'POST'])
def menu():
    if 'username' not in session:
        return redirect(url_for('login'))

    available_models = ['Modèle A', 'Modèle B']

    if request.method == 'POST':
        selected_model = request.form.get('model')
        file = request.files.get('file')
        segmentation_result, statistics = segmentation_process(selected_model)


        if not file or not allowed_file(file.filename):
            flash("Fichier invalide ou manquant !", 'error')
            return redirect(url_for('menu'))

            # Convertir l’image en mémoire (sans la sauvegarder)
        try:
            img = Image.open(BytesIO(file.read()))
        except Exception:
            flash("Erreur lors de la lecture de l'image.", 'error')
            return redirect(url_for('menu'))

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Enregistrer le résultat dans l'historique de l'utilisateur
        new_history = History(
            user_id=session.get('user_id'),
            model=selected_model,
            result=segmentation_result,
            surface_segmentee=statistics.get('surface_segmentee'),
            autre_mesure=statistics.get('autre_mesure')
        )
        db.session.add(new_history)
        db.session.commit()

        return render_template('result.html', result=segmentation_result, stats=statistics, img_data=img_base64)

    return render_template('menu.html', models=available_models)


# Fonction simulant la segmentation
def segmentation_process(model):
    segmentation_result = f"Résultat de segmentation pour le {model}"
    statistics = {
        "surface_segmentee": 123.45,  # Exemple de valeur
        "autre_mesure": 67.89  # Exemple de valeur
    }
    return segmentation_result, statistics


# Route pour afficher l'historique des segmentations de l'utilisateur connecté
@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    histories = History.query.join(User).filter(History.user_id == user_id).order_by(History.date.desc()).all()
    local_tz = pytz.timezone('Europe/Paris')
    for entry in histories:
        entry.date = entry.date.replace(tzinfo=pytz.utc).astimezone(local_tz)

    return render_template('history.html', histories=histories)


# Route de déconnexion
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    # Crée les tables dans la base de données si elles n'existent pas déjà
    with app.app_context():
        db.create_all()
    app.run(debug=True)
