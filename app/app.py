import os

from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pytz
from io import BytesIO
from PIL import Image
import base64
import torch
import segmentation_models_pytorch as smp
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask_migrate import Migrate

# Function to load the appropriate model based on the user's choice.
def load_model(model_type):
    if model_type == "Unet_resnet50":
        model = smp.Unet(
            encoder_name="resnet50",       # Using resnet50 encoder
            encoder_weights="imagenet",      # Pretrained on ImageNet
            in_channels=1,                   # Single channel (grayscale)
            classes=1                        # Binary segmentation
        )
        model.load_state_dict(torch.load("../unet_resnet50.pth", map_location=torch.device("cpu")))
        model.eval()
        return model
    elif model_type == "Unet_vgg16":
        model = smp.Unet(
            encoder_name="vgg16",          # Using vgg16 encoder
            encoder_weights="imagenet",      # Pretrained on ImageNet
            in_channels=1,                   # Single channel (grayscale)
            classes=1                        # Binary segmentation
        )
        model.load_state_dict(torch.load("../unet_vgg16.pth", map_location=torch.device("cpu")))
        model.eval()
        return model
    else:
        raise ValueError("Unknown model type: " + model_type)


def predict_mask(model, image):
    """
    Takes a model and a PIL image, preprocesses it (in grayscale),
    predicts the mask using segmentation_models_pytorch, overlays the mask
    (in transparent red) on the original image, and returns a segmentation
    result message and the composite image encoded in base64.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Convert image to grayscale
    image_gray = image.convert("L")
    image_np = np.array(image_gray)
    # Expand dims to create shape (H, W, 1)
    image_np = np.expand_dims(image_np, axis=-1)

    # Preprocess with albumentations
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0,), std=(1,)),
        ToTensorV2()
    ])
    transformed = transform(image=image_np)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        mask = (output > 0.5).float().cpu().numpy()[0, 0]

    # Convert mask to 0-255 uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Prepare original image for overlay: convert to RGB and resize
    original_rgb = image.convert("RGB").resize((256, 256))

    # Overlay function: applies the mask (colored in red with transparency) over the image
    def overlay_mask(original, mask, alpha=0.5, mask_color=(255, 0, 0)):
        colored_mask = Image.new("RGB", original.size, mask_color)
        alpha_mask = Image.fromarray(mask).point(lambda p: int(alpha * 255) if p > 0 else 0).convert("L")
        composite = original.copy()
        composite.paste(colored_mask, (0, 0), alpha_mask)
        return composite

    composite_image = overlay_mask(original_rgb, mask_uint8, alpha=0.5, mask_color=(255, 0, 0))

    buf = BytesIO()
    composite_image.save(buf, format="PNG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    segmentation_result = f"Segmentation réalisée avec le modèle {model.__class__.__name__}."
    return segmentation_result, img_base64



app = Flask(__name__)
app.secret_key = 'Projet_echo'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# User model
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


# History model with the new composite_image column
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model = db.Column(db.String(50))
    result = db.Column(db.Text)
    composite_image = db.Column(db.Text)  # Field to store the composite image as base64
    surface_segmentee = db.Column(db.Float)
    autre_mesure = db.Column(db.Float)
    date = db.Column(db.DateTime, default=datetime.utcnow)


@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('confirm')

        if password != confirm:
            flash('Les mots de passe ne correspondent pas.', 'error')
            return render_template('register.html')

        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Nom d\'utilisateur ou email déjà utilisé.', 'error')
            return render_template('register.html')

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Compte créé avec succès. Vous pouvez maintenant vous connecter.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['username'] = user.username
            session['user_id'] = user.id
            return redirect(url_for('menu'))
        else:
            flash("Nom d'utilisateur ou mot de passe incorrect.", 'error')
    return render_template('login.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/menu', methods=['GET', 'POST'])
def menu():
    if 'username' not in session:
        return redirect(url_for('login'))

    # Now the user can choose between two models.
    available_models = ['Unet_resnet50', 'Unet_vgg16']

    if request.method == 'POST':
        selected_model = request.form.get('model')
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            flash("Fichier invalide ou manquant !", 'error')
            return redirect(url_for('menu'))

        try:
            img = Image.open(BytesIO(file.read()))
        except Exception:
            flash("Erreur lors de la lecture de l'image.", 'error')
            return redirect(url_for('menu'))

        # Load the chosen model using the helper function
        try:
            model = load_model(selected_model)
        except ValueError as e:
            flash(str(e), 'error')
            return redirect(url_for('menu'))

        segmentation_result, mask_base64 = predict_mask(model, img)

        # Save to History including the composite image
        new_history = History(
            user_id=session.get('user_id'),
            model=selected_model,
            result=segmentation_result,
            composite_image=mask_base64
        )
        db.session.add(new_history)
        db.session.commit()

        return render_template('result.html', result=segmentation_result, img_data=mask_base64, model_name=selected_model)

    return render_template('menu.html', models=available_models)


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


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # In development; in production, consider using migrations.
    app.run(debug=True)
