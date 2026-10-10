# NOK Neural Network Lab

An interactive, small neural network for exploring the relationship between Norway's policy rate and the EUR/NOK exchange rate. The learning algorithm is written in NumPy; the interactive interface uses Streamlit. A separate script runs the same kind of experiment with plots and terminal output.

This is an educational classification exercise, not a currency forecast or evidence that interest rates cause exchange-rate movements.

## Requirements

- Python 3.10 or newer
- An internet connection for the first run, unless you provide the two CSV files described below

## Install

Clone the repository and open its directory:

```sh
git clone https://github.com/albertovth/nn-lab.git
cd nn-lab
```

Create a virtual environment and install the dependencies. On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r nok_network_requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r nok_network_requirements.txt
```

## Run

With the virtual environment active, start the interactive app:

```sh
python -m streamlit run nok_neural_network.py
```

Streamlit prints the local URL to open in a browser. The app lets you inspect the monthly data, the decision boundary, individual neurons, and the training process. Its controls change the network size, learning rate, training length, and displayed epoch.

For a smaller, standalone experiment instead:

```sh
python nok_neural_network_minimal.py
```

The standalone script prints the later-period accuracy and a simple baseline, then opens two Matplotlib figures. A graphical desktop is needed to display those figures.

## Data

The programs use daily policy-rate and EUR/NOK observations republished from Norges Bank data by [frefrik/norway-finance-statistics](https://github.com/frefrik/norway-finance-statistics). The original data can be explored through [Norges Bank's data portal](https://app.norges-bank.no/query/).

The programs download these CSV files when they are absent:

- [Policy rate: `no_keyPolicyRate.csv`](https://raw.githubusercontent.com/frefrik/norway-finance-statistics/master/data/no_keyPolicyRate.csv)
- [Exchange rates: `no_exchangeRates.csv`](https://raw.githubusercontent.com/frefrik/norway-finance-statistics/master/data/no_exchangeRates.csv)

For offline use, download both files and place them beside the Python scripts. Local copies take precedence over the URLs. The downloaded data are not included in this repository; their terms are separate from this project's MIT license.

## What the model does

Daily observations are averaged by calendar month. For month *t*, the two inputs are the average policy rate and the percentage change in monthly-average EUR/NOK from the previous month. The target is whether monthly-average EUR/NOK rises in month *t + 1* (a weaker NOK against the euro). The current, incomplete month and months without complete inputs and target are excluded.

The network has two inputs, one ReLU hidden layer, and a sigmoid output. It is trained with full-batch gradient descent and binary cross-entropy. The earlier 80% of usable months are used for training; the later 20% are held out for comparison. Input scaling is estimated from the training period only. The interactive app compares its later-period accuracy with a majority-class baseline.

The dataset is small, the inputs omit many influences on exchange rates, and historical accuracy can vary with the period and settings. The results should not be used for trading or policy decisions.

## License

The code is available under the [MIT License](LICENSE). The source CSV files have their own provenance and terms.
