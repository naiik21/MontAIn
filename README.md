# MontAIn

Proyecto con frontend en Astro/React y API en Python para analizar rutas de montaña.

## Estructura del repositorio

```text
MONTAIN/
├── front/              → frontend (desplegar en Vercel)
├── api/                → backend/API (desplegar en Render/Railway)
│   ├── main.py
│   ├── requirements.txt
│   ├── models/
│   └── .env            → NO se commitea
├── data/               → datos locales (gpx, etc.) NO se commitean si son pesados
├── .gitignore
└── README.md
```

## Backend (api/)

```bash
cd api
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

En producción (Render/Railway):
- Sube solo la carpeta `api/`.
- Configura las variables de entorno (`.env`) desde el panel de la plataforma.

## Frontend (front/)

```bash
cd front
pnpm install
pnpm dev
```

En producción:
- Conecta el repo a Vercel y selecciona `front/` como root del proyecto.