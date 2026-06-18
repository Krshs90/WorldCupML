# WorldCupML 🏆

**Made by Krshs90**  
*ReadMe By Gemini 3.1*  
*Bugfixed by Google Antigravity*

---

## 📌 About the Project
WorldCupML is an AI-powered desktop application built in Python that accurately predicts FIFA World Cup match outcomes. It fetches live tournament data and compares it against decades of historical match results, running complex machine learning algorithms and millions of mathematical simulations to forecast match winners and exact scorelines.

## ✨ Features
- **Real-Time Match Data:** Pulls live tournament fixtures, current live scores, detailed team lineups (starting XI vs subs), and group standings straight from the ESPN API.
- **XGBoost Machine Learning:** Dynamically calculates Expected Goals (xG) by engineering advanced statistical features like Elo ratings, recent team form, and venue context.
- **Monte Carlo Simulations:** Runs up to 100 million Poisson-distributed parallel universe match scenarios in milliseconds using vectorized Numpy arrays to calculate precise probabilities.
- **Premium UI/UX:** A sleek, beautiful, fully responsive dark-mode GUI engineered with CustomTkinter.

## 🚀 Download & Play
You don't need to be a programmer or have Python installed to use WorldCupML! 

1. Head over to the **[Releases Tab](https://github.com/Krshs90/WorldCupML/releases)** on this repository.
2. Download the latest `WorldCupML.exe` file from the V1.0 Release.
3. Double click it to launch the app instantly!

## 💻 For Developers
If you'd like to run the code from the source, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Krshs90/WorldCupML.git
cd WorldCupML
pip install -r requirements.txt
python app.py
```

## 🛠️ Tech Stack
- **Python** (Core Language)
- **CustomTkinter** (Frontend Interface)
- **XGBoost & Scikit-Learn** (Predictive ML Regressors)
- **Pandas & Numpy** (Data Analytics & Simulation Engine)
- **ESPN API** (Live backend payload)
