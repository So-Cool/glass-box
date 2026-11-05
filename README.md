# Glass-Box

Glass-Box explains the decisions of a loan-approval decision tree through a
conversation (talking to the device or typing in a chat window).
An applicant -- one of ten predefined personas -- asks *why* their application
was declined, whether the decision is *fair* and *what they can do*
to change it, and Glass-Box answers with counterfactual statements, e.g.,:

> If you had requested your loan for at most 11 or between 23 and 43 months,
> rather than 15 months, your loan would be accepted.

The hardware implementation of Glass-Box (demonstrated at IJCAI 2018)
is a voice-enabled device based on Raspberry Pi, Google AIY Voice Kit,
LED displays and a QR code scanner.
The Glass-Box software can also be run as a text-only chat on any computer.
The loan decisions come from a decision tree trained on
the [UCI German Credit][gc] data;
the counterfactuals are produced by [CtreeX][ctreex].

A summary of bug fixes and implementation changes introduced when modernising
the code is provided in the [changelog](./CHANGELOG.md#001).

## ToC

* [Installation](#installation-with-uv)
* [Quick start](#quick-start)
* [Talking to Glass-Box](#talking-to-glass-box)
* [Commands (`cmd:`)](#commands-cmd)
* [Personas and features](#personas-and-features)
    * [Personas](#personas)
    * [Features](#features)
* [Configuration](#configuration)
* [Command line tool](#command-line-tool)
* [Natural language understanding](#natural-language-understanding)
* [Data and model](#data-and-model)
* [Web application and device server](#web-application-and-device-server)
* [Glass-Box device](#glass-box-device)
    * [Components, wiring and software](#components-wiring-and-software)
    * [Software architecture](#software-architecture)
    * [Set-up on (modern) Raspberry Pi OS](#set-up-on-modern-raspberry-pi-os)
* [Development](#development)
* [Citation](#citation)

## Installation with uv

```sh
git clone https://github.com/So-Cool/glass-box.git
cd glass-box
uv sync                              # locked environment with test dependencies
```

Optional installation extras:

* `uv sync --extra data` -- download and pre-process the German Credit data
  with [MLaaS][mlaas] (the core Glass-Box functionality does not require this);
* `uv sync --extra pi` -- hardware drivers for the Raspberry Pi (Linux only);
* `uv sync --group notebooks` -- example Jupyter Notebook execution.

## Quick start

Chat in the terminal:

```sh
uv run glassbox chat
```

or in the browser (http://127.0.0.1:5000):

```sh
uv run glassbox -c config/laptop.toml serve
```

For the Python library use install the necessary dependencies and
see the Jupyter Notebook examples in `notebooks`:

```sh
uv sync --group notebooks
uv run jupyter lab notebooks/
```

| Notebook               | Content |
|------------------------|---------|
| `01_dialogue`          | model, dialogue, language understanding and assembled system |
| `02_data_and_model`    | data (with checks), original tree, retraining |
| `03_original_vs_fixed` | effects of code fixes |

## Talking to Glass-Box

An example conversation (with abridged answers):

* Hello.
  > Hello. Welcome to Robo Bank. If you would like to apply for a loan please
  > provide your name or application number. [...]
* My application number is 3.
  > Thank you for your application, Ruth. [...]
  > Would you like to know the outcome?
* What is the outcome?
  > Your application has been declined.
* Why?
  > [...] If you had asked for at most 883 or between 955 and 1181 or
  > between 1286 and 1917 or between 2111 and 2578 pounds,
  > rather than 3959 pounds, your loan would be accepted.
* Is the decision fair?
  > I believe that this decision is indirectly unfair. [...]
  > if you had been male (divorced or separated) or male (single) or male
  > (married or widowed), rather than being female
  > (divorced or separated or married).
* Disregarding my loan amount, what can I do to get my loan accepted?
  > [...] The outcome of your application would change if you had
  > requested your loan for at most 11 or between 23 and 43 months,
  > rather than 15 months. (When avoiding using: loan amount.)
* What is my age?
  > Dear Ruth, at the time of application: you have been 29 years old.
* Tell me about myself.
  > Dear Ruth, at the time of application: you have requested your loan for
  > 15 months and you have asked for 3959 pounds and [...]
* cmd: why->features_drop:[A2,A5]
  > [...] Your loan would be accepted had you had less than 200 pounds on your
  > current account or no current account, instead of having overdraft on your
  > current account. (When avoiding using: loan duration and loan amount.)
* Goodbye!
  > Thank you for your application. Goodbye.

A conversation starts with a greeting, continues with an application
(a persona's name or number, or a scanned application card) and then the
applicant can ask any of the questions below in any order.
Asking the same question again gives the next explanation
(the next counterfactual);
when all the explanations have been given Glass-Box says so, and the next
request starts from the beginning.
The wording of the answers is chosen at random from a few sentence templates.

| Request                           | Example sentences | Answer |
|-----------------------------------|-------------------|--------|
| greeting (`hello`)                | "hello", "good morning" | welcome and how to apply |
| application (`apply`)             | "my application number is 3", "I am Ruth", "it's four" | processes the application of a persona |
| outcome (`outcome`)               | "what is the outcome", "was my loan approved" | accepted or declined |
| explanation (`why`)               | "why", "why was my loan rejected", "explain the decision" | the next counterfactual explanation |
| restricted explanation (`why`)    | "why, given my age", "why despite my marital status", "why given my age despite my gender", "explain the decision disregarding my loan amount" | explanations that must use (*given*, *considering*, *taking into account*) or must not use (*despite*, *disregarding*, *ignoring*, *regardless of*) the features mentioned |
| restart (`why`)                   | "restart why", "reset the explanations" | the explanations are communicated from the beginning |
| fairness (`fairness`)             | "is the decision fair", "was I discriminated against" | counterfactuals involving the protected features (marital status and gender, migrant status); "indirectly unfair" when they need other changes too |
| what to do (`action`)             | "what can I do to get my loan accepted", "what steps can I take" | counterfactuals that change only *actionable* features |
| restricted suggestions (`action`) | "disregarding my age, what can I do to get my loan accepted", "what can I do with my instalment rate to get my loan accepted" | suggestions without (or with) the features mentioned |
| about me (`myself`)               | "tell me about myself", "what is my age", "tell me about my savings account and my job" | the applicant's details: the features mentioned or, without any, all of them in the chat and only the current account status when speaking (to keep spoken answers short) |
| help (`help`)                     | "help", "what should I ask" | example questions |
| repeat (`repeat`)                 | "repeat", "say that again" | the last answer |
| goodbye (`bye`)                   | "goodbye", "that is all, thanks" | ends the application |

There are also two special messages:

* `disable` mutes (and unmutes) spoken answers, and
* `quit` / `exit` do nothing -- they are refused and the device keeps running
  (for the next applicant).

The sentences are understood by a statistical model trained on examples
(see [Natural language understanding](#natural-language-understanding));
when a question is misunderstood, rephrasing it or the `cmd:` syntax below
helps.

## Commands (`cmd:`)

Any request can be issued directly, bypassing the language understanding:

```text
cmd: <intent>
cmd: <intent>-><argument>:<value>;<argument>:[<value>,<value>]
```

Values in square brackets are lists;
`true` / `false` are booleans;
features are given by their ids (see [Features](#features)).

| Command    | Arguments | Example |
|------------|-----------|---------|
| `hello`    | --                                    | `cmd: hello`                      |
| `apply`    | `persona` + id `00`-`09`              | `cmd: apply->persona:04`          |
| `outcome`  | --                                    | `cmd: outcome`                    |
| `why`      | `features_require` for features every explanation must change and `features_drop` for features no explanation may change; `restart` starts from the first explanation | `cmd: why->features_require:[A13];features_drop:[A9]`, `cmd: why->restart:true` |
| `fairness` | --                                    | `cmd: fairness`                   |
| `action`   | `features_require`, `features_drop`   | `cmd: action->features_drop:[A5]` |
| `myself`   | `features` for the features to report | `cmd: myself->features:[A13,A9]`  |
| `help`     | --                                    | `cmd: help`                       |
| `repeat`   | --                                    | `cmd: repeat`                     |
| `bye`      | --                                    | `cmd: bye`                        |

Changing the restrictions of `why` (or `action`) starts a new list of
explanations;
repeating the same command gives the next one.

## Personas and features

### Personas

| Id  | Name       | Decision | Source                 |
|-----|------------|----------|------------------------|
| 00  | Stu        | declined | row 54 of the UCI data |
| 01  | Jonathan   | declined | synthetic              |
| 02  | Lurleen    | declined | row 783                |
| 03  | Ruth       | declined | row 858                |
| 04  | Joe        | declined | row 472                |
| 05  | Kirk       | accepted | row 60                 |
| 06  | Mona       | accepted | synthetic              |
| 07  | Selma      | declined | synthetic              |
| 08  | Montgomery | accepted | synthetic              |
| 09  | Marge      | accepted | synthetic              |

Rows are 0-based rows of `german.data`;
the personas are stored in `src/glassbox/resources/personas.json` and a test
checks the copied ones against the data set.

### Features

The decision tree uses 13 features of the German Credit data;
their description is stored in
`src/glassbox/resources/german_credit_features.json`.
*Actionable* features are those an applicant can change ("what can I do");
*protected* features are those a fair decision should not depend on.

| Id  | Feature                         | Type        | Spec       |
|-----|---------------------------------|-------------|------------|
| A1  | current account status          | categorical | actionable |
| A2  | loan duration (months)          | numerical   | actionable |
| A3  | credit history                  | categorical |            |
| A5  | loan amount (pounds)            | numerical   | actionable |
| A6  | savings account status          | categorical | actionable |
| A7  | current employment duration     | categorical |            |
| A8  | instalment rate (percent)       | numerical   | actionable |
| A9  | marital status and gender       | categorical | protected  |
| A13 | age                             | numerical   |            |
| A14 | other loans                     | categorical | actionable |
| A15 | housing type                    | categorical |            |
| A17 | job type                        | categorical | actionable |
| A20 | migrant status (foreign worker) | categorical | protected  |

## Configuration

`glassbox -c <file.toml> ...` reads a TOML file (also `$GLASSBOX_CONFIG`);
every value has a default (`glassbox config` prints the effective
configuration).
Example config files are in `config/`:

| File                           | Use                                                                              |
|--------------------------------|----------------------------------------------------------------------------------|
| `laptop.toml`                  | chat on a laptop; LEDs are logged, no speech                                     |
| `laptop-with-device.toml`      | the original set-up: the application on a laptop, the hardware on a Raspberry Pi |
| `raspberry-pi-device.toml`     | the device server on the Raspberry Pi                                            |
| `raspberry-pi-standalone.toml` | everything on the Raspberry Pi                                                   |

The following options can be configured:

| Section             | Keys |
|---------------------|------|
| `[app]`             | `host`, `port`, `poll_interval` (seconds between polls for voice and QR messages) |
| `[model]`           | `tree` points to a tree as JSON or a (modern) scikit-learn pickle; default: the original tree |
| `[dialogue]`        | `implementation` (`original` / `fixed` explainer); `[dialogue.explainer_fixes]` for individual CtreeX fixes |
| `[nlu]`             | `backend` (`spacy` / `rasa-legacy`), `thinking_delay` (seconds the "thinking" LEDs are shown); `[nlu.spacy]` has `model_dir`, `spacy_model`; `[nlu.rasa_legacy]` has `url`, `project`, `model` |
| `[hardware.leds]`   | `backend` (`neopixel` / `remote` / `mock` / `disabled`), `url` (remote), `ring` / `square` (neopixel options, e.g., `led_dma`, `brightness`) |
| `[hardware.speech]` | `backend` (`aiy` / `remote` / `mock` / `disabled`), `url`, `lang`, `volume` |
| `[hardware.voice]`  | `enabled`, `app_url` for push-to-talk voice input (Raspberry Pi) |
| `[hardware.qr]`     | `enabled`, `app_url`, `device` for QR code scanner (Raspberry Pi) |
| `[device]`          | `host`, `port`; `[device.leds]`, `[device.speech]` for the device server's hardware (default: `[hardware.*]`) |

## Command line tool

| Command | Purpose |
|---------|---------|
| `glassbox chat [--implementation original]`                        | chat in the terminal |
| `glassbox serve [--host H] [--port P] [--implementation original]` | the web chat (and the endpoints of the voice input and the QR scanner) |
| `glassbox device [--host H] [--port P]`                            | the Raspberry Pi device server (LEDs, speech, voice input, QR scanner) |
| `glassbox data [--data-home DIR]`                                  | download the German Credit data, pre-process it and verify it against the original data |
| `glassbox nlu train [--spacy-model M] [--output DIR] [--seed N]`   | retrain the default NLU model on the original and the added training data |
| `glassbox nlu parse "sentence"`                                    | show how a sentence is understood |
| `glassbox nlu generate [--repeats N] [--seed N] [--test-repeats N] [--test-seed N] [--output FILE] [--test-output FILE]` | generate the added NLU training data and the held-out test set from templates (the defaults reproduce the shipped files) |
| `glassbox nlu test [--test-set FILE] [--seed N]`                   | train on the original and on all the training data and evaluate both on the test set |
| `glassbox model retrain --output tree.json [--random-state N]`     | train a new tree on the German Credit data |
| `glassbox model export --input tree.pkl --output tree.json`        | convert a scikit-learn tree to JSON |
| `glassbox config`                                                  | print the effective configuration |

Global options:

* `-c/--config FILE`
* `-v/--verbose`

## Natural language understanding

Two backends turn sentences into requests
(they can be selected with the `[nlu] backend` option in the config file):

* `spacy` (default) reimplements the original Rasa NLU 0.12 pipeline with
  spaCy (`en_core_web_sm`), a conditional random field for the entities and
  a linear SVM for the intents;
  a pipeline trained like this is shipped with the package
  (the conditional random field in `crf.joblib`, the linear SVM in
  `intent.joblib` and the metadata in `metadata.json`, all in the
  `src/glassbox/resources/nlu/model/` directory);
  `glassbox nlu train` retrains it.

* `rasa-legacy` (set `nlu.backend = "rasa-legacy"` and `nlu.rasa_legacy.url`)
  serves the *original* model based on Rasa NLU 0.12.3, which is
  compatible with Python up to version 3.6, and available through Docker
  (`docker/rasa-legacy`);

  ```sh
  docker build -t glassbox-rasa-legacy docker/rasa-legacy
  docker run --rm -p 5005:5005 glassbox-rasa-legacy

  curl -XPOST 127.0.0.1:5005/parse -d '{"q": "hello", "project": "default"}'
  ```

  the `docker/rasa-legacy/projects/default/chat` directory holds the original
  pipeline files (`crf_model.pkl`, `intent_classifier_sklearn.pkl` and
  `metadata.json`).

The original training data consisted of 162 examples;
these can be found in
`docker/rasa-legacy/projects/default/chat/training_data.json`,
with the format adapted to the new pipeline in
`src/glassbox/resources/nlu/training_data.json`.
To improve the accuracy of the new (`spacy`) pipeline, 241 examples
(generated from templates by `glassbox nlu generate`)
were added in 2026 and can be found in the
`src/glassbox/resources/nlu/training_data_extra.json` file.
On 49 held-out sentences -- generated with different templates by the same
command and stored in `tests/data/nlu_test_set.json` --
the accuracy (`glassbox nlu test`) improved as follows:

|                              | original data | original + new data |
|------------------------------|---------------|---------------------|
| intent accuracy              | 38.8%         | 81.6%               |
| entity recall                | 53.8%         | 84.6%               |
| entity precision             | 87.5%         | 97.1%               |
| (intent, arguments) accuracy | 24.5%         | 73.5%               |

Note that the additional examples were specifically generated to improve
performance on the categories with large errors so these numbers are optimistic.

## Data and model

No raw data are stored in this repository.
The model is the original decision tree
(`DecisionTreeClassifier(min_samples_split=3)` with 194 leaves),
converted from its scikit-learn 0.19 pickle to JSON and stored in
`src/glassbox/resources/glassbox_german_credit_tree.json`.
Working with the data (retraining, notebooks, data tests) requires
the `data` extra (`uv sync --extra data`).
Executing

```sh
uv run glassbox data
```

downloads `german.data` from UCI (cached in `~/.cache/mlaas` or `$MLAAS_DATA`),
pre-processes it to match the original data used by Glass-Box and
checks it against the SHA-256 checksum of the data on which the original tree
was trained;
it also checks the five personas copied from the data set.

## Web application and device server

`glassbox serve` provides the following endpoints (port 5000):

| Endpoint                        | Purpose |
|---------------------------------|---------|
| `GET /`                         | the chat page |
| `POST /ask`                     | `{"message": ...}` (or form field `messageText`) -> `{"answer": ...}` |
| `POST /voice` (`/voicehat`)     | voice input: `{"text": ...}` -> `{"text": <answer to speak>}` |
| `POST /qrcode`                  | QR scanner: `{"persona": "04"}` or `{"qr": "QR-Code:04"}` |
| `GET /messages` (`/getmessage`) | messages from the voice input and the QR scanner for the chat page |
| `GET /health`                   | health check |

`glassbox device` provides the following endpoints on the Raspberry Pi
(port 5001):

| Endpoint                 | Purpose |
|--------------------------|---------|
| `POST /led`              | an LED command |
| `POST /say`  (`/aiysay`) | voice output: `{"text": ...}` |
| `GET /health`            | health check |

The names in brackets are the original endpoints.

## Glass-Box device

Glass-Box hardware is a Raspberry Pi with
a Google AIY Voice Kit v1 (microphone, speaker, push button),
two WS281x LED displays (a 16-LED ring and an 8x8-LED "square" matrix) and
a camera reading the QR codes printed on the application cards.
None of the hardware is needed to run Glass-Box on a laptop;
to this end, use the `mock` or `disabled` backends configured in `[hardware.*]`.
`config/laptop.toml` delivers chat only with the behaviour of the LEDs logged.

### Components, wiring and software

| Component | Connection | Software |
|-----------|------------|----------|
| Google AIY Voice Kit v1 (Voice HAT, button, microphone, speaker) | HAT on the GPIO header | `aiy` library (`glassbox.hardware.speech`, `glassbox.hardware.voice`) |
| 16-LED WS2812 ring                                               | Voice HAT *Servo 4* = GPIO 12, PWM channel 0, DMA 11 | `neopixel_python.ring.RingLED` (`glassbox.hardware.leds`) |
| 8x8-LED WS2812 "square" matrix                                   | Voice HAT *Servo 2* = GPIO 13, PWM channel 1, DMA 11 | `neopixel_python.square.SquareLED` (`glassbox.hardware.leds`) |
| Raspberry Pi camera                                              | CSI connector, `/dev/video0` via `bcm2835-v4l2` | `zbarcam` (`glassbox.hardware.qr`) |

The pins are the defaults of [`neopixel_python`][neopixel] version 0.1.0;
DMA channel 11 is set in the configuration (`ring = { led_dma = 11 }`).
The LED displays need a 5 V power supply able to deliver their current;
share the ground with the Pi.

### Software architecture

```text
 Raspberry Pi                                    laptop (or the Pi itself)
 ┌───────────────────────────────┐   HTTP        ┌──────────────────────────┐
 │ glassbox device               │ ───────────▶  │ glassbox serve           │
 │  voice input  -> POST /voice  │               │  chat page, dialogue,    │
 │  QR scanner   -> POST /qrcode │  ◀─────────── │  NLU, model              │
 │  POST /led, POST /say         │   LEDs/speech │                          │
 └───────────────────────────────┘               └──────────────────────────┘
```

The original set-up ran the application on a laptop, with the remaining
hardware functionality on the Raspberry Pi;
this can be replicated with the `config/raspberry-pi-device.toml` and
`config/laptop-with-device.toml` configuration files.
Alternatively, both processes can be run on the Raspberry Pi with the
`config/raspberry-pi-standalone.toml` configuration file.

### Set-up on (modern) Raspberry Pi OS

The original Raspberry Pi setup is described in
the [changelog](./CHANGELOG.md#pre-001).
Below is this process adapted for modern Raspberry Pi OS;
**note that it has not been tested yet**.

1. Flash *Raspberry Pi OS (64-bit) Bookworm* (Python 3.11) and enable SSH and
   the camera (`sudo raspi-config` -> *Interface Options*).
   On Bookworm the camera uses libcamera;
   for `/dev/video0` add `dtoverlay=` for your camera module or
   use `libcamerify zbarcam ...`.

2. Install system packages:

   ```sh
   sudo apt-get update
   sudo apt-get install -y git zbar-tools alsa-utils libttspico-utils \
       python3-dev build-essential scons swig
   ```

3. Enable Voice HAT sound card in `/boot/firmware/config.txt`:

   ```ini
   #dtparam=audio=on
   dtparam=i2s=on
   dtoverlay=googlevoicehat-soundcard
   ```

4. Set up AIY library (text-to-speech, button, speech recognition).
   Google's current `aiyprojects-raspbian` repository supports
   Voice Kit v1 and v2:

   ```sh
   git clone https://github.com/google/aiyprojects-raspbian.git ~/AIY-projects-python
   ```

   It is installed into the Glass-Box environment in step 5.
   Since Google Assistant gRPC API was deprecated, this new version falls back
   to **Google Cloud Speech** (`aiy.cloudspeech`), which needs a Google Cloud
   project with the Speech-to-Text API and a service-account key
   (`~/cloud_speech.json`, see the AIY documentation).
   Text-to-speech (`aiy.voice.tts`) works offline with `pico2wave`.

5. Install uv and Glass-Box:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   git clone https://github.com/So-Cool/glass-box.git && cd glass-box
   uv sync --extra pi                         # LED drivers (rpi_ws281x)
   uv pip install -e ~/AIY-projects-python    # the aiy package
   ```

6. Run (the LED drivers need root to access the PWM/DMA hardware):

   ```sh
   sudo .venv/bin/glassbox -c config/raspberry-pi-device.toml device
   ```

   and on the laptop
   `uv run glassbox -c config/laptop-with-device.toml serve`.
   Adjust `app_url` (the laptop) and the LED/speech `url` (the Pi) in
   the two config files.
   To start the device server at boot, create a systemd unit running
   the command above.

## Development

```sh
uv run pytest                             # tests

uv run glassbox data                      # download and process German Credit

uv run glassbox model retrain --output tree.json   # train a new tree
uv run glassbox model export --input tree.pkl --output tree.json
                                          # convert a scikit-learn tree to JSON

uv run glassbox nlu generate              # regenerate the added NLU data
uv run glassbox nlu train                 # retrain shipped NLU model
uv run glassbox nlu test                  # evaluate the NLU
uv run glassbox nlu parse "txt"           # show how txt is understood
```

## Citation

```bibtex
@inproceedings{sokol2018glass,
  title={{Glass-Box}: {Explaining} {AI} decisions with counterfactual
         statements through conversation with a voice-enabled virtual
         assistant},
  author={Sokol, Kacper and Flach, Peter},
  booktitle={Proceedings of the 27th International Joint Conference on
             Artificial Intelligence: Demos},
  publisher={IJCAI Organization},
  pages={5868--5870},
  year={2018},
  doi={10.24963/ijcai.2018/865}
}
```

[gc]: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
[ctreex]: https://github.com/So-Cool/ctreex
[uv]: https://docs.astral.sh/uv/
[mlaas]: https://github.com/So-Cool/MLaaS
[neopixel]: https://github.com/So-Cool/neopixel-python
