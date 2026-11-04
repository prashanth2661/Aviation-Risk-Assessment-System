import joblib
import pandas as pd
import os
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config

class RiskEngine:
    def __init__(self):
        self.preprocessor = joblib.load(os.path.join(Config.MODELS_DIR, "preprocessor.pkl"))
        self.model = joblib.load(os.path.join(Config.MODELS_DIR, "ensemble", "stacking_model.pkl"))
        self.le = joblib.load(os.path.join(Config.MODELS_DIR, "label_encoder.pkl"))
        
        try:
            self.feature_names = self.preprocessor.feature_names_in_
        except AttributeError:
            raw_df = pd.read_csv(Config.RAW_DATA)
            self.feature_names = raw_df.drop(columns=['risk_level']).columns.tolist()

    def _generate_pilot_assistance(self, risk, prob, data):
        """Generates real-world cockpit callouts and actions."""
        callouts = []
        actions = []
        
        v_speed = data.get('vertical_speed_fpm', 0)
        alt_agl = data.get('altitude_agl_ft', 500)
        
        if v_speed < -1000:
            callouts.append("SINK RATE, SINK RATE!")
            actions.append("Reduce vertical speed immediately.")
        
        pitch = data.get('pitch_attitude_deg', 0)
        roll = data.get('roll_attitude_deg', 0)
        if abs(roll) > 20:
            callouts.append("BANK ANGLE!")
            actions.append("Level the wings.")
        if pitch > 10 or pitch < -2:
            callouts.append("PITCH ATTITUDE!")
            
        if data.get('gpws_alert') not in [None, 'None', np.nan]:
            callouts.append(f"GPWS: {data['gpws_alert'].upper()}!")
        if data.get('tcas_alert') not in [None, 'None', np.nan]:
            callouts.append(f"TCAS: {data['tcas_alert'].upper()}!")

        if risk == "High":
            primary_action = "GO-AROUND"
            detail = "UNSTABLE APPROACH. Execute missed approach procedure. Climb to missed approach altitude."
        elif risk == "Medium":
            primary_action = "CAUTION"
            detail = "APPROACH SEMI-STABLE. Correct flight path and airspeed. Monitor closely."
        else:
            primary_action = "STABLE"
            detail = "Continue approach. Clear to land."

        return {
            "action": primary_action,
            "callouts": " | ".join(callouts) if callouts else "MINIMA, CHECKED",
            "detail": detail,
            "checklist": actions if actions else ["Maintain glide path", "Verify landing configuration"]
        }

    def predict(self, input_dict):
        df = pd.DataFrame([input_dict])
        
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        
        df = df[self.feature_names]
        processed = self.preprocessor.transform(df)
        
        idx = self.model.predict(processed)[0]
        risk = self.le.inverse_transform([idx])[0]
        prob = self.model.predict_proba(processed)[0][idx]
        
        assistance = self._generate_pilot_assistance(risk, prob, input_dict)
        
        return risk, prob, assistance