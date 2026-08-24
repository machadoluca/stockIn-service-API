from flask import Flask, request, jsonify
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
import traceback

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "API de previsão de séries temporais usando Prophet."

@app.route("/forecast", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "trainning_data" not in data:
            return jsonify({"error": "Campo 'trainning_data' é obrigatório."}), 400
        
        df_train = pd.DataFrame(data["trainning_data"])

        df_train["ds"] = pd.to_datetime(df_train["ds"])
        df_train["y"] = pd.to_numeric(df_train["y"])

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )

        model.fit(df_train)

        horizon = data.get("horizon", 7)

        future = model.make_future_dataframe(
            periods=horizon,
            freq="D"
        )

        forecast = model.predict(future)

        initial = data.get("initial", "60 days")
        period = data.get("period", "7 days")
        horizon_str = data.get("horizon_str", "7 days")

        try:
            df_cv = cross_validation(
                model,
                initial=initial,
                period=period,
                horizon=horizon_str
            )

            df_perf = performance_metrics(df_cv)
            metrics_mean = df_perf.mean().to_dict()

        except Exception as cv_error:
            metrics_mean = {
                "error": "Falha ao executar cross-validation.",
                "details": str(cv_error)
            }

        for k, v in metrics_mean.items():
            if isinstance(v, pd.Timedelta):
                metrics_mean[k] = str(v)

        last_train_date = df_train["ds"].max()
        forecast_future = forecast[forecast["ds"] > last_train_date]

        return jsonify({
            "forecast": forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records"),
            "metrics": metrics_mean
        })

    except Exception as e:
        return jsonify({
            "error": "Erro interno no servidor Flask.",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
