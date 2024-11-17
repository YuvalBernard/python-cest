"""
Module that defines the app structure
"""

import os
import sys

import arviz as az
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyqtwaitingspinner import SpinnerParameters, WaitingSpinner

from fit_spectra import bayesian_mcmc, bayesian_vi, least_squares
from simulate_spectra import (
    batch_gen_spectrum_analytical,
    batch_gen_spectrum_numerical,
    batch_gen_spectrum_symbolic,
)

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOption("antialias", True)


class DataPage(QWizardPage):
    def __init__(self):
        super().__init__()

        self.setTitle("Select Data")
        page_label = QLabel(
            "Select .xlsx file with data to fit.\n"
            "Each column should be attributed to a different saturation power (in µT)."
        )
        page_label.setWordWrap(True)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(page_label)

        open_button = QPushButton("Open file")
        open_button.clicked.connect(self.get_filename)
        self.label = QLabel("None selected.")

        file_select_layout = QHBoxLayout()
        file_select_layout.addWidget(open_button)
        file_select_layout.addWidget(self.label)
        self.layout.addLayout(file_select_layout)

        self.table = pg.TableWidget(editable=False, sortable=False)
        self.table.setAlternatingRowColors(True)

        self.graphWidget = pg.PlotWidget()
        self.graphWidget.invertX()
        self.graphWidget.setMouseEnabled(False, False)
        self.graphWidget.setLabel("left", "Z-value [a.u.]")
        self.graphWidget.setLabel("bottom", "offset [ppm]")

    def get_filename(self):
        self.filename, extension = QFileDialog.getOpenFileName(self)
        self.label.setText(self.filename)
        if self.filename:
            self.wizard().data = pd.read_excel(self.filename)
            self.table.setData(self.wizard().data.T.to_dict())

            plot_and_table_layout = QHBoxLayout()
            plot_and_table_layout.addWidget(self.table, alignment=Qt.AlignmentFlag.AlignCenter)
            plot_and_table_layout.addWidget(
                self.graphWidget, alignment=Qt.AlignmentFlag.AlignCenter
            )
            self.graphWidget.clear()
            powers = list(
                self.wizard().data.columns[1:].str.extract(r"([-+]?\d*\.?\d+)", expand=False)
            )
            for i in range(n := self.wizard().data.shape[1] - 1):
                self.graphWidget.addLegend(offset=(1, -1))
                self.graphWidget.plot(
                    self.wizard().data.iloc[:, 0],
                    self.wizard().data.iloc[:, i + 1],
                    pen=(i, n),
                    name=f"B₁ = {float(powers[i]):.2g} µT",
                )

            self.layout.addLayout(plot_and_table_layout)


class ConstantsGroup(QGroupBox):
    def __init__(self, parent: QWizardPage):
        super().__init__(parent)

        self.setTitle("Constants")
        layout = QGridLayout(self)

        b0_label = QLabel("B₀ (T)")
        gamma_label = QLabel("γ (MHz/T)")
        tp_label = QLabel("tₚ (s)")
        b1_label = QLabel("B₁ (µT)")

        b0_desc = QLabel("External magnetic field strength")
        gamma_desc = QLabel("Gyromagnetic ratio")
        tp_desc = QLabel("Saturation pulse duration")
        b1_desc = QLabel("Saturation pulse amplitude\n(comma separated list)")

        b0_entry = QLineEdit()
        gamma_entry = QLineEdit()
        tp_entry = QLineEdit()
        self.b1_entry = QLineEdit(placeholderText="e.g. 1.0, 3.0, 5.0")

        parent.registerField("b0", b0_entry)
        parent.registerField("gamma", gamma_entry)
        parent.registerField("tp", tp_entry)
        parent.registerField("b1*", self.b1_entry)

        layout.addWidget(makeBold(QLabel("Parameter")), 0, 0)
        layout.addWidget(makeBold(QLabel("Value")), 0, 1)
        layout.addWidget(makeBold(QLabel("Description")), 0, 2)
        layout.addWidget(b0_label, 1, 0, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(b0_entry, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(b0_desc, 1, 2, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(gamma_label, 2, 0, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(gamma_entry, 2, 1, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(gamma_desc, 2, 2, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(tp_label, 3, 0, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(tp_entry, 3, 1, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(tp_desc, 3, 2, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(b1_label, 4, 0, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.b1_entry, 4, 1, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(b1_desc, 4, 2, alignment=Qt.AlignmentFlag.AlignTop)

        button = QPushButton("Fill B₁ from table")
        button.clicked.connect(lambda: self.fill_in_b1_list(parent))
        layout.addWidget(button, 5, 0, 1, -1)

        # For testing purposes:
        b0_entry.setText("9.4")
        gamma_entry.setText("42.58")
        tp_entry.setText("10")

    def fill_in_b1_list(self, parent):
        if hasattr(parent.wizard(), "data"):
            df = parent.wizard().data
            # if the excel sheet headers are in µT:
            if df.columns[1:].str.contains("T").any():
                b1_list = list(df.columns[1:].str.extract(r"([-+]?\d*\.?\d+)", expand=False))
                self.b1_entry.setText(", ".join([f"{float(b1):.2f}" for b1 in b1_list]))
            elif df.columns[1:].str.contains("Hz").any():
                b1_list = list(df.columns[1:].str.extract(r"([-+]?\d*\.?\d+)", expand=False))
                try:
                    gamma = float(parent.field("gamma"))
                    self.b1_entry.setText(", ".join([f"{float(b1)/gamma:.2f}" for b1 in b1_list]))
                except ValueError:
                    QMessageBox.warning(self, "Info", "Please insert gyromagnetic ratio.")

        else:
            QMessageBox.warning(self, "Warning", "Please select data excel sheet on previous page.")


class VariablesGroup(QGroupBox):
    def __init__(self, parent: QWizardPage):
        super().__init__(parent)

        self.setTitle("Variables")
        layout = QGridLayout(self)

        R1a_label = QLabel("R1a (Hz)")
        R2a_label = QLabel("R2a (Hz)")
        dwa_label = QLabel("dwa (ppm)")
        R1b_label = QLabel("R1b (Hz)")
        R2b_label = QLabel("R2b (Hz)")
        kb_label = QLabel("kb (Hz)")
        fb_label = QLabel("fb (Hz)")
        dwb_label = QLabel("dwb (ppm)")

        R1a_desc = QLabel("Pool A longitudinal relaxation rate")
        R2a_desc = QLabel("Pool A transverse relaxation rate")
        dwa_desc = QLabel("Larmor frequency of pool A relative to itself. Optimally zero")
        R1b_desc = QLabel("Pool B longitudinal relaxation rate")
        R2b_desc = QLabel("Pool B transverse relaxation rate")
        kb_desc = QLabel("Pool B forward exchange rate")
        fb_desc = QLabel("Pool B equilibrium magnetization; fraction relative to pool A")
        dwb_desc = QLabel("Larmor frequency of pool B relative to pool A")

        R1a_vary = QComboBox()
        R2a_vary = QComboBox()
        dwa_vary = QComboBox()
        self.R1b_vary = QComboBox()
        R2b_vary = QComboBox()
        kb_vary = QComboBox()
        fb_vary = QComboBox()
        dwb_vary = QComboBox()

        for combo_box, name in zip(
            [R1a_vary, R2a_vary, dwa_vary, self.R1b_vary, R2b_vary, kb_vary, fb_vary, dwb_vary],
            [
                "R1a_vary",
                "R2a_vary",
                "dwa_vary",
                "R1b_vary",
                "R2b_vary",
                "kb_vary",
                "fb_vary",
                "dwb_vary",
            ],
        ):
            combo_box.addItems(["Vary", "Static"])
            parent.registerField(name, combo_box)

        R1a_val = QLineEdit()
        R2a_val = QLineEdit()
        dwa_val = QLineEdit()
        self.R1b_val = QLineEdit()
        R2b_val = QLineEdit()
        kb_val = QLineEdit()
        fb_val = QLineEdit()
        dwb_val = QLineEdit()

        R1a_init_val = QLineEdit()
        R2a_init_val = QLineEdit()
        dwa_init_val = QLineEdit()
        R1b_init_val = QLineEdit()
        R2b_init_val = QLineEdit()
        kb_init_val = QLineEdit()
        fb_init_val = QLineEdit()
        dwb_init_val = QLineEdit()

        R1a_min = QLineEdit()
        R2a_min = QLineEdit()
        dwa_min = QLineEdit()
        R1b_min = QLineEdit()
        R2b_min = QLineEdit()
        kb_min = QLineEdit()
        fb_min = QLineEdit()
        dwb_min = QLineEdit()

        R1a_max = QLineEdit()
        R2a_max = QLineEdit()
        dwa_max = QLineEdit()
        R1b_max = QLineEdit()
        R2b_max = QLineEdit()
        kb_max = QLineEdit()
        fb_max = QLineEdit()
        dwb_max = QLineEdit()

        layout.addWidget(makeBold(QLabel("Parameter")), 0, 0)
        layout.addWidget(makeBold(QLabel("State")), 0, 1)
        layout.addWidget(makeBold(QLabel("Min")), 0, 2)
        layout.addWidget(makeBold(QLabel("Max")), 0, 3)
        layout.addWidget(makeBold(QLabel("Init Value")), 0, 4)
        layout.addWidget(makeBold(QLabel("Static Value")), 0, 5)
        layout.addWidget(makeBold(QLabel("Description")), 0, 6)

        # Create registerFields to track values
        for i, (
            label,
            combo_box,
            min,
            max,
            val,
            init_val,
            desc,
            name_min,
            name_max,
            name_val,
            name_init_val,
        ) in enumerate(
            zip(
                [
                    R1a_label,
                    R2a_label,
                    dwa_label,
                    R1b_label,
                    R2b_label,
                    kb_label,
                    fb_label,
                    dwb_label,
                ],
                [R1a_vary, R2a_vary, dwa_vary, self.R1b_vary, R2b_vary, kb_vary, fb_vary, dwb_vary],
                [R1a_min, R2a_min, dwa_min, R1b_min, R2b_min, kb_min, fb_min, dwb_min],
                [R1a_max, R2a_max, dwa_max, R1b_max, R2b_max, kb_max, fb_max, dwb_max],
                [R1a_val, R2a_val, dwa_val, self.R1b_val, R2b_val, kb_val, fb_val, dwb_val],
                [
                    R1a_init_val,
                    R2a_init_val,
                    dwa_init_val,
                    R1b_init_val,
                    R2b_init_val,
                    kb_init_val,
                    fb_init_val,
                    dwb_init_val,
                ],
                [R1a_desc, R2a_desc, dwa_desc, R1b_desc, R2b_desc, kb_desc, fb_desc, dwb_desc],
                [
                    "R1a_min",
                    "R2a_min",
                    "dwa_min",
                    "R1b_min",
                    "R2b_min",
                    "kb_min",
                    "fb_min",
                    "dwb_min",
                ],
                [
                    "R1a_max",
                    "R2a_max",
                    "dwa_max",
                    "R1b_max",
                    "R2b_max",
                    "kb_max",
                    "fb_max",
                    "dwb_max",
                ],
                [
                    "R1a_val",
                    "R2a_val",
                    "dwa_val",
                    "R1b_val",
                    "R2b_val",
                    "kb_val",
                    "fb_val",
                    "dwb_val",
                ],
                [
                    "R1a_init_val",
                    "R2a_init_val",
                    "dwa_init_val",
                    "R1b_init_val",
                    "R2b_init_val",
                    "kb_init_val",
                    "fb_init_val",
                    "dwb_init_val",
                ],
            )
        ):
            combo_box.currentIndexChanged.connect(min.setDisabled)
            combo_box.currentIndexChanged.connect(max.setDisabled)
            combo_box.currentIndexChanged.connect(init_val.setDisabled)
            combo_box.currentIndexChanged.connect(val.setEnabled)

            parent.registerField(name_min, min)
            parent.registerField(name_max, max)
            parent.registerField(name_init_val, init_val)
            parent.registerField(name_val, val)

            if combo_box.currentText() == "Vary":
                min.setEnabled(True)
                max.setEnabled(True)
                init_val.setEnabled(True)
                val.setDisabled(True)
            else:
                min.setDisabled(True)
                max.setDisabled(True)
                init_val.setDisabled(True)
                val.setEnabled(True)

            j = i + 1
            layout.addWidget(label, j, 0)
            layout.addWidget(combo_box, j, 1)
            layout.addWidget(min, j, 2)
            layout.addWidget(max, j, 3)
            layout.addWidget(init_val, j, 4)
            layout.addWidget(val, j, 5)
            layout.addWidget(desc, j, 6)

        # For testing purposes:
        R1a_vary.setCurrentText("Static")
        R2a_vary.setCurrentText("Static")
        dwa_vary.setCurrentText("Static")
        R1a_val.setText("0.33")
        R2a_val.setText("0.5")
        dwa_val.setText("0")
        R1b_min.setText("0.1")
        R1b_max.setText("10")
        R1b_init_val.setText("5")
        R2b_min.setText("0.1")
        R2b_max.setText("100")
        R2b_init_val.setText("1")
        kb_min.setText("10")
        kb_max.setText("500")
        kb_init_val.setText("150")
        fb_min.setText("1e-5")
        fb_max.setText("5e-3")
        fb_init_val.setText("1e-3")
        dwb_min.setText("3")
        dwb_max.setText("4")
        dwb_init_val.setText("3.5")


class SolverGroup(QGroupBox):
    def __init__(self, parent: QWizardPage):
        super().__init__(parent)

        self.setTitle("Solver and Fitting Method")
        layout = QVBoxLayout(self)

        solver_list = QComboBox()
        solver_list.addItems(["Symbolic", "Analytical", "Numerical"])
        parent.registerField("solver", solver_list)
        solver_list.currentIndexChanged.connect(lambda: self.validateSolver(parent))

        fitter_list = QComboBox()
        fitter_list.addItems(["Bayesian, MCMC", "Bayesian, ADVI", "Nonlinear Least Squares"])
        parent.registerField("fitting_method", fitter_list)

        selection_layout = QFormLayout()
        selection_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        selection_layout.addRow("Solver:", solver_list)
        selection_layout.addRow("Fitting method:", fitter_list)
        layout.addLayout(selection_layout)
        layout.addSpacing(20)
        layout.addWidget(makeBold(QLabel("Configure fitting method")))

        config_page = QStackedLayout()
        layout.addLayout(config_page)
        fitter_list.currentIndexChanged.connect(config_page.setCurrentIndex)

        # Configure MCMC widget
        bayesian_mcmc_container = QWidget()
        bayesian_mcmc_layout = QFormLayout(bayesian_mcmc_container)
        bayesian_mcmc_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )

        warmup_entry = QLineEdit("1000")
        samples_entry = QLineEdit("1000")
        chains_entry = QLineEdit("4")

        parent.registerField("num_warmup", warmup_entry)
        parent.registerField("num_samples", samples_entry)
        parent.registerField("num_chains", chains_entry)

        bayesian_mcmc_layout.addRow("Number of warmup samples", warmup_entry)
        bayesian_mcmc_layout.addRow("Number of posterior samples", samples_entry)
        bayesian_mcmc_layout.addRow("Number of chains", chains_entry)
        config_page.addWidget(bayesian_mcmc_container)

        # Configure ADVI widget
        bayesian_vi_container = QWidget()
        bayesian_vi_layout = QFormLayout(bayesian_vi_container)
        bayesian_vi_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        stepsize_entry = QLineEdit("1e-3")
        steps_entry = QLineEdit("80_000")
        posterior_samples_entry = QLineEdit("4000")

        parent.registerField("optimizer_stepsize", stepsize_entry)
        parent.registerField("optimizer_num_steps", steps_entry)
        parent.registerField("num_posterior_samples", posterior_samples_entry)

        bayesian_vi_layout.addRow("Optimizer stepsize:", stepsize_entry)
        bayesian_vi_layout.addRow("Number of optimization steps:", steps_entry)
        bayesian_vi_layout.addRow("Number of posterior samples:", posterior_samples_entry)

        config_page.addWidget(bayesian_vi_container)

        least_squares_container = QLabel("Please fill in initial values for varying parameters.")
        config_page.addWidget(least_squares_container)

    def validateSolver(self, parent):
        if parent.field("solver") == 1:  # Analytical
            parent.variablesGroup.R1b_vary.setCurrentText("Static")
            parent.variablesGroup.R1b_vary.removeItem(0)
            parent.variablesGroup.R1b_val.setText("0")
            QMessageBox.information(
                self, "Info", "Using the Analytical solver disables R1b for fitting!"
            )
        else:
            parent.variablesGroup.R1b_vary.clear()
            parent.variablesGroup.R1b_vary.addItems(["Vary", "Static"])


class ModelPage(QWizardPage):
    def __init__(self, parent: QWizard):
        super().__init__(parent)

        self.setTitle("Configure Model Parameters")
        layout = QVBoxLayout(self)
        layout2 = QHBoxLayout()
        layout2.addWidget(ConstantsGroup(self))
        layout2.addWidget(SolverGroup(self))
        layout.addLayout(layout2)
        self.variablesGroup = VariablesGroup(self)
        layout.addWidget(self.variablesGroup)

        self.setCommitPage(True)
        parent.button(QWizard.WizardButton.CommitButton).clicked.connect(self.makeModel)

    def makeModel(self):
        self.wizard().model_parameters = lmfit.Parameters()

        for name, init, vary, min, max, val in zip(
            ["R1a", "R2a", "dwa", "R1b", "R2b", "kb", "fb", "dwb"],
            [
                "R1a_init_val",
                "R2a_init_val",
                "dwa_init_val",
                "R1b_init_val",
                "R2b_init_val",
                "kb_init_val",
                "fb_init_val",
                "dwb_init_val",
            ],
            [
                "R1a_vary",
                "R2a_vary",
                "dwa_vary",
                "R1b_vary",
                "R2b_vary",
                "kb_vary",
                "fb_vary",
                "dwb_vary",
            ],
            ["R1a_min", "R2a_min", "dwa_min", "R1b_min", "R2b_min", "kb_min", "fb_min", "dwb_min"],
            ["R1a_max", "R2a_max", "dwa_max", "R1b_max", "R2b_max", "kb_max", "fb_max", "dwb_max"],
            ["R1a_val", "R2a_val", "dwa_val", "R1b_val", "R2b_val", "kb_val", "fb_val", "dwb_val"],
        ):
            par_varies = not self.field(vary)  # field returns 0 if parameter varies...
            if par_varies:
                par_min = float(self.field(min))
                par_max = float(self.field(max))
            else:
                par_min = None
                par_max = None
            try:
                par_value = float(self.field(init) if par_varies else self.field(val))
            except ValueError:  # field probably empty. Choose average as init value
                par_value = (min + max) / 2

            self.wizard().model_parameters.add(
                name=name, value=par_value, vary=par_varies, min=par_min, max=par_max
            )


class Thread(QThread):
    result = pyqtSignal(dict)

    def __init__(self, fitting_method, args):
        super().__init__()
        self.fitting_method = fitting_method
        self.args = args

    @pyqtSlot()
    def run(self):
        self.result.emit(self.fitting_method(*self.args))


class ResultPage(QWizardPage):
    def __init__(self, parent: QWizard):
        super().__init__(parent)

        spin_pars = SpinnerParameters(disable_parent_when_spinning=True)
        self.spinner = WaitingSpinner(self, spin_pars)

        self.setTitle("Performing Fitting... Please Wait for the Process to Finish")
        page_layout = QVBoxLayout(self)
        page_layout.addWidget(self.spinner)
        parent.button(QWizard.WizardButton.CommitButton).clicked.connect(self.perform_fit)

        sub_page_layout = QHBoxLayout()

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        font = self.summary.font()
        font.setPointSize(11)
        self.summary.setFont(font)

        self.graphWidget = MatplotlibWidget()
        sub_page_layout.addWidget(self.summary)
        sub_page_layout.addStretch()
        sub_page_layout.addWidget(self.graphWidget)
        self.graphWidget.setVisible(False)
        page_layout.addLayout(sub_page_layout)

        self.button = QPushButton("Save Results")
        page_layout.addWidget(self.button)
        self.button.clicked.connect(self.save_results)
        self.button.setVisible(False)

    def perform_fit(self):
        df = self.wizard().data
        b0 = float(self.field("b0"))
        gamma = float(self.field("gamma")) * 2 * np.pi
        tp = float(self.field("tp"))
        self.powers = np.array([float(b1) for b1 in self.field("b1").split(",")])
        self.offsets = df.to_numpy(dtype=float).T[0]
        self.data = df.to_numpy(dtype=float).T[1:]
        model_parameters = self.wizard().model_parameters
        self.model_args = (self.offsets, self.powers, b0, gamma, tp)

        match self.field("solver"):
            case 0:  # "Symbolic"
                self.solver = batch_gen_spectrum_symbolic
            case 1:  # "Analytical":
                self.solver = batch_gen_spectrum_analytical
            case 2:  # "Numerical"
                self.solver = batch_gen_spectrum_numerical

        match self.field("fitting_method"):
            case 0:  # "Bayesian, MCMC"
                fitting_method = bayesian_mcmc
                num_warmup = int(n_warmup) if (n_warmup := self.field("num_warmup")) else None
                num_samples = int(n_samples) if (n_samples := self.field("num_samples")) else None
                num_chains = int(n_chains) if (n_chains := self.field("num_chains")) else None
                args = (
                    model_parameters,
                    self.model_args,
                    self.data,
                    self.solver,
                    num_warmup,
                    num_samples,
                    num_chains,
                )
            case 1:  # "Bayesian, ADVI"
                fitting_method = bayesian_vi
                optimizer_step_size = (
                    float(step_size) if (step_size := self.field("optimizer_stepsize")) else None
                )
                optimizer_num_steps = (
                    int(n_steps) if (n_steps := self.field("optimizer_num_steps")) else None
                )
                num_posterior_samples = (
                    int(n_samples) if (n_samples := self.field("num_posterior_samples")) else None
                )
                args = (
                    model_parameters,
                    self.model_args,
                    self.data,
                    self.solver,
                    optimizer_step_size,
                    optimizer_num_steps,
                    num_posterior_samples,
                )
            case 2:  # "Nonlinear Least Squares"
                fitting_method = least_squares
                args = (model_parameters, self.model_args, self.data, self.solver)

        # Do this in a separate thread!
        self.thread = Thread(fitting_method, args)
        self.thread.result.connect(self.summarize_fit)
        self.thread.start()
        self.spinner.start()

    def summarize_fit(self, result):
        self.fit_result = result
        self.spinner.stop()
        self.setTitle("Fit Summary")

        match self.field("fitting_method"):
            case 0:  # MCMC
                mcmc = result["fit"]
                self.idata = az.from_numpyro(mcmc)
                self.fit_summary = az.summary(
                    self.idata,
                    kind="stats",
                    round_to=4,
                    var_names=["~sigma"],
                    stat_funcs={"median": np.median},
                )
                self.summary.setText(self.print_results())
                best_fit_pars = tuple(
                    np.median(mcmc.get_samples()[par])
                    if self.wizard().model_parameters[par].vary
                    else self.wizard().model_parameters[par].value
                    for par in self.wizard().model_parameters.keys()
                )
            case 1:  # ADVI
                posterior_samples = result["fit"]
                self.svi_losses = result["loss"]
                self.idata = az.from_dict(posterior_samples)
                self.fit_summary = az.summary(
                    self.idata,
                    kind="stats",
                    round_to=4,
                    var_names=["~sigma"],
                    stat_funcs={"median": np.median},
                )
                self.summary.setText(self.print_results())
                best_fit_pars = tuple(
                    np.median(posterior_samples[par])
                    if self.wizard().model_parameters[par].vary
                    else self.wizard().model_parameters[par].value
                    for par in self.wizard().model_parameters.keys()
                )
            case 2:  # NLS
                fit_params = result["fit"]
                self.summary.setText(lmfit.fit_report(fit_params))
                best_fit_pars = tuple(fit_params.valuesdict().values())

        self.fit = self.solver(best_fit_pars, *self.model_args).T

        self.fig = self.graphWidget.getFigure()
        self.fig.set_layout_engine("tight")
        ax = self.fig.add_subplot()
        ax.plot(
            self.offsets,
            self.data.T,
            marker=".",
            lw=0,
            label=[f"B₁={power:.1f} μT" for power in self.powers],
        )
        ax.set_prop_cycle(None)
        ax.plot(self.offsets, self.fit)
        ax.set_title(
            "Nonlinear Least Squares Fit"
            if self.field("fitting_method") == 2
            else "Bayesian Fit, Posterior Median"
        )
        ax.set_xlabel("offset [ppm]")
        ax.set_ylabel("Z-value [a.u.]")
        ax.invert_xaxis()
        ax.legend()
        self.graphWidget.setVisible(True)
        self.button.setVisible(True)

    def print_results(self):
        buff = []
        add = buff.append
        parnames = self.wizard().model_parameters.keys()
        namelen = max(len(n) for n in parnames)

        params = self.fit_summary.T

        add("[[Variables]]")
        for name in parnames:
            space = " " * (namelen - len(name))
            nout = f"{name}:{space}"
            if name in params.columns:
                par = params[name]
                sval = f"{par["median"]:.7g}"
                serr = f"{par["sd"]:.7g}"
                try:
                    spercent = f"({abs(par["sd"]/par["median"]):.2%})"
                except ZeroDivisionError:
                    spercent = ""
                sval = f"{sval} +/- {serr} {spercent}"
                add(f"    {nout} {sval}")
            else:
                par = self.wizard().model_parameters[name]
                add(f"    {nout} {par.value: .7g} (fixed)")

        return "\n".join(buff)

    def save_results(self):
        folderPath = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folderPath:
            return
        fitting_methods = ["Bayesian-MCMC", "Bayesian-ADVI", "Least-Squares"]
        solvers = ["Symbolic", "Analytical", "Numerical"]
        saveDir = os.path.join(
            folderPath, fitting_methods[self.field("fitting_method")], solvers[self.field("solver")]
        )
        os.makedirs(saveDir, exist_ok=True)

        self.fig.savefig(os.path.join(saveDir, "best_fit.pdf"), format="pdf")
        with pd.ExcelWriter(os.path.join(saveDir, "best_fit.xlsx"), mode="w") as writer:
            self.wizard().data.to_excel(writer, sheet_name="data", index=False)

            match self.field("fitting_method"):
                case 0:  # MCMC
                    posterior_samples = self.fit_result["fit"].get_samples()
                    self.best_fit_pars_mean = np.asarray(
                        [
                            np.mean(posterior_samples[par])
                            if self.wizard().model_parameters[par].vary
                            else self.wizard().model_parameters[par].value
                            for par in self.wizard().model_parameters.keys()
                        ]
                    )
                    self.best_fit_pars_median = np.asarray(
                        [
                            np.median(posterior_samples[par])
                            if self.wizard().model_parameters[par].vary
                            else self.wizard().model_parameters[par].value
                            for par in self.wizard().model_parameters.keys()
                        ]
                    )
                    self.best_fit_pars_mode = np.asarray(
                        [
                            az.plots.plot_utils.calculate_point_estimate(
                                "mode", posterior_samples[par]
                            )
                            if self.wizard().model_parameters[par].vary
                            else self.wizard().model_parameters[par].value
                            for par in self.wizard().model_parameters.keys()
                        ]
                    )
                    self.best_fit_spectra_mean = self.solver(
                        self.best_fit_pars_mean, *self.model_args
                    ).T
                    self.best_fit_spectra_median = self.solver(
                        self.best_fit_pars_median, *self.model_args
                    ).T
                    self.best_fit_spectra_mode = self.solver(
                        self.best_fit_pars_mode, *self.model_args
                    ).T

                    pd.DataFrame(
                        np.c_[self.offsets, self.best_fit_spectra_mean],
                        columns=["ppm"] + [f"{power:.1f} μT" for power in self.powers],
                    ).round(3).to_excel(writer, sheet_name="mean", index=False)
                    pd.DataFrame(
                        np.c_[self.offsets, self.best_fit_spectra_median],
                        columns=["ppm"] + [f"{power:.1f} μT" for power in self.powers],
                    ).round(3).to_excel(writer, sheet_name="median", index=False)
                    pd.DataFrame(
                        np.c_[self.offsets, self.best_fit_spectra_mode],
                        columns=["ppm"] + [f"{power:.1f} μT" for power in self.powers],
                    ).round(3).to_excel(writer, sheet_name="mode", index=False)

                    az.plot_pair(
                        self.idata,
                        var_names=["~sigma"],
                        kind="kde",
                        marginals=True,
                        divergences=True,
                    )[0, 0].get_figure().savefig(
                        os.path.join(saveDir, "pair_plot.pdf"), format="pdf"
                    )
                    az.plot_ess(self.idata, var_names=["~sigma"], relative=True)[
                        0, 0
                    ].get_figure().savefig(os.path.join(saveDir, "ess_plot.pdf"), format="pdf")
                    with open(os.path.join(saveDir, "fit_summary.txt"), "w") as file:
                        file.write("[Fixed Variables]\n")
                        for par in self.wizard().model_parameters.keys():
                            if not self.wizard().model_parameters[par].vary:
                                file.write(f"{par}\t{self.wizard().model_parameters[par].value}\n")
                        file.write("\n[Fit Variables]\n")
                        file.write(
                            az.summary(
                                self.idata,
                                round_to=4,
                                kind="stats",
                                stat_funcs={
                                    "median": np.median,
                                    "mode": lambda x: az.plots.plot_utils.calculate_point_estimate(
                                        "mode", x
                                    ),
                                },
                            ).to_string()
                        )
                        file.write("\n[Fit Diagnostics]\n")
                        fit_diagnostics = az.summary(
                            self.idata,
                            kind="diagnostics",
                            round_to=4,
                        )
                        file.write(fit_diagnostics.to_string())
                case 1:  # ADVI
                    posterior_samples = self.fit_result["fit"]
                    self.best_fit_pars_mean = np.asarray(
                        [
                            np.mean(posterior_samples[par])
                            if self.wizard().model_parameters[par].vary
                            else self.wizard().model_parameters[par].value
                            for par in self.wizard().model_parameters.keys()
                        ]
                    )
                    self.best_fit_pars_median = np.asarray(
                        [
                            np.median(posterior_samples[par])
                            if self.wizard().model_parameters[par].vary
                            else self.wizard().model_parameters[par].value
                            for par in self.wizard().model_parameters.keys()
                        ]
                    )
                    self.best_fit_pars_mode = np.asarray(
                        [
                            az.plots.plot_utils.calculate_point_estimate(
                                "mode", posterior_samples[par]
                            )
                            if self.wizard().model_parameters[par].vary
                            else self.wizard().model_parameters[par].value
                            for par in self.wizard().model_parameters.keys()
                        ]
                    )
                    self.best_fit_spectra_mean = self.solver(
                        self.best_fit_pars_mean, *self.model_args
                    ).T
                    self.best_fit_spectra_median = self.solver(
                        self.best_fit_pars_median, *self.model_args
                    ).T
                    self.best_fit_spectra_mode = self.solver(
                        self.best_fit_pars_mode, *self.model_args
                    ).T

                    pd.DataFrame(
                        np.c_[self.offsets, self.best_fit_spectra_mean],
                        columns=["ppm"] + [f"{power:.1f} μT" for power in self.powers],
                    ).round(3).to_excel(writer, sheet_name="mean", index=False)
                    pd.DataFrame(
                        np.c_[self.offsets, self.best_fit_spectra_median],
                        columns=["ppm"] + [f"{power:.1f} μT" for power in self.powers],
                    ).round(3).to_excel(writer, sheet_name="median", index=False)
                    pd.DataFrame(
                        np.c_[self.offsets, self.best_fit_spectra_mode],
                        columns=["ppm"] + [f"{power:.1f} μT" for power in self.powers],
                    ).round(3).to_excel(writer, sheet_name="mode", index=False)

                    az.plot_pair(self.idata, var_names=["~sigma"], kind="kde", marginals=True)[
                        0, 0
                    ].get_figure().savefig(os.path.join(saveDir, "pair_plot.pdf"), format="pdf")
                    fig, ax = plt.subplots()
                    ax.plot(self.svi_losses)
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("ELBO")
                    ax.set_title("SVI Loss per iteration")
                    fig.savefig(os.path.join(saveDir, "svi_loss.pdf"), format="pdf")

                    with open(os.path.join(saveDir, "fit_summary.txt"), "w") as file:
                        file.write("[Fixed Variables]\n")
                        for par in self.wizard().model_parameters.keys():
                            if not self.wizard().model_parameters[par].vary:
                                file.write(f"{par}\t{self.wizard().model_parameters[par].value}\n")
                        file.write("\n[Fit Variables]\n")
                        file.write(
                            az.summary(
                                self.idata,
                                round_to=4,
                                kind="stats",
                                stat_funcs={
                                    "median": np.median,
                                    "mode": lambda x: az.plots.plot_utils.calculate_point_estimate(
                                        "mode", x
                                    ),
                                },
                            ).to_string()
                        )
                case 2:  # NLS
                    df_fit = pd.DataFrame(
                        np.c_[self.offsets, self.fit],
                        columns=["ppm"] + [f"{power:.2g} μT" for power in self.powers],
                    )
                    df_fit.round(3).to_excel(writer, sheet_name="nls", index=False)

                    with open(os.path.join(saveDir, "fit_summary.txt"), "w") as file:
                        file.write(lmfit.fit_report(self.fit_result["fit"]))


class WizardApp(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Bloch-McConnellizer")
        self.setMinimumSize(QSize(940, 640))
        self.addPage(DataPage())
        self.addPage(ModelPage(self))
        self.addPage(ResultPage(self))


def makeBold(widget: QWidget):
    f = widget.font()
    f.setWeight(QFont.Weight.DemiBold)
    widget.setFont(f)
    return widget


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WizardApp()
    window.show()
    sys.exit(app.exec())
