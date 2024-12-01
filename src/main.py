"""
Module that defines the app structure
"""

import os
import sys
from pathlib import Path

import arviz as az
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
import toml
from PyQt6.QtCore import QObject, QRunnable, QSize, Qt, QThread, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
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
from pyqtwaitingspinner import WaitingSpinner

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
            "Each column should be attributed to a different saturation power (in µT or Hz)."
        )
        page_label.setWordWrap(True)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(page_label)

        open_button = QPushButton("Open file")
        open_button.clicked.connect(self.get_filename)
        self.label = QLabel("None selected.")

        self.normalize_button = QPushButton("Normalize Data")
        self.normalize_button.clicked.connect(self.normalizeData)

        file_select_layout = QHBoxLayout()
        file_select_layout.addWidget(open_button)
        file_select_layout.addWidget(self.label)
        self.layout.addLayout(file_select_layout)
        self.layout.addWidget(self.normalize_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.normalize_button.setVisible(False)

        self.table = pg.TableWidget(editable=False, sortable=False)
        self.table.setAlternatingRowColors(True)

        self.graphWidget = pg.PlotWidget()
        self.graphWidget.invertX()
        self.graphWidget.setMouseEnabled(False, False)
        self.graphWidget.setLabel("left", "Z-value [a.u.]")
        self.graphWidget.setLabel("bottom", "offset [ppm]")

    def normalizeData(self):
        self.wizard().data = pd.read_excel(self.wizard().filename)
        selected_option, ok = QInputDialog.getItem(
            self,
            "Normalize Data",
            "Select method to normalize data",
            ["By First Value", "By Maximal Value"],
            current=0,
            editable=False,
        )
        if ok:
            if selected_option == "By First Value":
                for col in self.wizard().data.columns[1:]:
                    self.wizard().data[col] /= self.wizard().data[col][0]
            else:
                for col in self.wizard().data.columns[1:]:
                    self.wizard().data[col] /= self.wizard().data[col].max()
            self.updateUi()
        else:
            return

    def updateUi(self):
        self.table.setData(self.wizard().data.T.to_dict())

        plot_and_table_layout = QHBoxLayout()
        plot_and_table_layout.addWidget(self.table, alignment=Qt.AlignmentFlag.AlignCenter)
        plot_and_table_layout.addWidget(self.graphWidget, alignment=Qt.AlignmentFlag.AlignCenter)
        self.graphWidget.clear()
        powers = self.wizard().data.columns[1:]
        for i in range(n := self.wizard().data.shape[1] - 1):
            self.graphWidget.addLegend(offset=(1, -1))
            self.graphWidget.plot(
                self.wizard().data.iloc[:, 0],
                self.wizard().data.iloc[:, i + 1],
                pen=(i, n),
                name=f"B₁ = {powers[i]}",
            )

        self.layout.addLayout(plot_and_table_layout)

    def get_filename(self):
        self.wizard().filename, extension = QFileDialog.getOpenFileName(self)
        self.label.setText(self.wizard().filename)
        if self.wizard().filename:
            try:
                self.wizard().data = pd.read_excel(self.wizard().filename)
                if (
                    self.wizard().data.iloc[:, 1:].max(axis=None) > 2
                ):  # arbitrary value that is impossible if data is normalized
                    QMessageBox.warning(self, "Warning", "Please normalize the data.")
                    self.normalize_button.setVisible(True)
            except ValueError:
                QMessageBox.warning(self, "Error", "The program only accepts data in xlsx format.")
                return
            self.updateUi()


class ConstantsGroup(QGroupBox):
    def __init__(self, parent: QWizardPage):
        super().__init__(parent)

        self.setTitle("Constants")
        self.parent = parent
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
        b1_entry = QLineEdit(placeholderText="e.g. 1.0, 3.0, 5.0")

        self.parent.registerField("b0", b0_entry)
        self.parent.registerField("gamma", gamma_entry)
        self.parent.registerField("tp", tp_entry)
        self.parent.registerField("b1*", b1_entry)

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
        layout.addWidget(b1_entry, 4, 1, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(b1_desc, 4, 2, alignment=Qt.AlignmentFlag.AlignTop)

        button = QPushButton("Fill B₁ from table")
        button.clicked.connect(self.fill_in_b1_list)
        layout.addWidget(button, 5, 0, 1, -1)

    def fill_in_b1_list(self):
        if hasattr(self.parent.wizard(), "data"):
            df = self.parent.wizard().data
            # if the excel sheet headers are in µT:
            if df.columns[1:].str.contains("T").any():
                b1_list = list(df.columns[1:].str.extract(r"([-+]?\d*\.?\d+)", expand=False))
                self.parent.setField("b1", ", ".join([f"{float(b1):.2f}" for b1 in b1_list]))
            elif df.columns[1:].str.contains("Hz").any():
                b1_list = list(df.columns[1:].str.extract(r"([-+]?\d*\.?\d+)", expand=False))
                try:
                    gamma = float(self.parent.field("gamma"))
                    self.parent.setField(
                        "b1", ", ".join([f"{float(b1)/gamma:.2f}" for b1 in b1_list])
                    )
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
        fb_label = QLabel("fb")
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
            parent.registerField(
                name,
                combo_box,
                "currentText",
                QComboBox.currentTextChanged,
            )

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


class SolverGroup(QGroupBox):
    def __init__(self, parent: QWizardPage):
        super().__init__(parent)

        self.parent = parent
        self.setTitle("Solver and Fitting Method")
        layout = QVBoxLayout(self)

        solver_list = QComboBox()
        solver_list.addItems(["Symbolic", "Analytical", "Numerical"])
        self.parent.registerField(
            "solver", solver_list, "currentText", QComboBox.currentTextChanged
        )
        solver_list.currentIndexChanged.connect(self.validateSolver)

        fitter_list = QComboBox()
        fitter_list.addItems(["Bayesian, MCMC", "Bayesian, ADVI", "Nonlinear Least Squares"])
        self.parent.registerField(
            "fitting_method", fitter_list, "currentText", QComboBox.currentTextChanged
        )

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

        self.parent.registerField("num_warmup", warmup_entry)
        self.parent.registerField("num_samples", samples_entry)
        self.parent.registerField("num_chains", chains_entry)

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

        self.parent.registerField("optimizer_stepsize", stepsize_entry)
        self.parent.registerField("optimizer_num_steps", steps_entry)
        self.parent.registerField("num_posterior_samples", posterior_samples_entry)

        bayesian_vi_layout.addRow("Optimizer stepsize:", stepsize_entry)
        bayesian_vi_layout.addRow("Number of optimization steps:", steps_entry)
        bayesian_vi_layout.addRow("Number of posterior samples:", posterior_samples_entry)

        config_page.addWidget(bayesian_vi_container)

        least_squares_container = QWidget()
        least_squares_layout = QFormLayout(least_squares_container)
        least_squares_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )

        algorithm_entry = QComboBox()
        algorithm_entry.addItems(
            [
                "Levenberg-Marquardt",
                "Trust Region Reflective",
                "Basin-Hopping",
                "Adaptive Memory Programming for Global Optimization",
            ]
        )
        self.parent.registerField(
            "NLS_algorithm",
            algorithm_entry,
            "currentText",
            QComboBox.currentTextChanged,
        )
        least_squares_layout.addRow("Algorithm", algorithm_entry)

        config_page.addWidget(least_squares_container)

    def validateSolver(self):
        if self.parent.field("solver") == "Analytical":  # Analytical
            self.parent.setField("R1b_val", 1)
            self.parent.setField("R1b_vary", "Static")
            self.parent.variablesGroup.R1b_vary.setEnabled(False)
            QMessageBox.information(
                self, "Info", "Using the Analytical solver disables R1b for fitting!"
            )
        else:
            self.parent.variablesGroup.R1b_vary.setEnabled(True)
            self.parent.setField("R1b_vary", "Vary")


class ModelPage(QWizardPage):
    def __init__(self, parent: QWizard):
        super().__init__(parent)

        self.setTitle("Configure Model Parameters")

        button = QPushButton("Load configuration file")
        button.clicked.connect(self.load_configuration)

        self.variablesGroup = VariablesGroup(self)
        self.constantsGroup = ConstantsGroup(self)
        self.solverGroup = SolverGroup(self)

        layout = QVBoxLayout(self)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout2 = QHBoxLayout()
        layout2.addWidget(self.constantsGroup)
        layout2.addWidget(self.solverGroup)
        layout.addLayout(layout2)
        layout.addWidget(self.variablesGroup)

    def validatePage(self):
        return super().validatePage() and self.makeModel()

    def load_configuration(self):
        filename, extension = QFileDialog.getOpenFileName(self)
        if filename:
            with open(filename, "r") as f:
                try:
                    config = toml.load(f)
                except UnicodeDecodeError:
                    QMessageBox.warning(self, "Error", "Config file must be of type toml.")
                    return

            self.setField("b0", config["Constants"]["B0"])
            self.setField("gamma", config["Constants"]["gamma"])
            self.setField("tp", config["Constants"]["tp"])

            for par, init, vary, min, max, val in zip(
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
            ):
                if config["Variables"][par]["vary"]:
                    self.setField(vary, "Vary")
                    self.setField(min, config["Variables"][par]["min"])
                    self.setField(max, config["Variables"][par]["max"])
                    self.setField(init, config["Variables"][par]["init"])
                else:
                    self.setField(vary, "Static")
                    self.setField(val, config["Variables"][par]["value"])

    def makeModel(self) -> bool:
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
            par_varies = True if self.field(vary) == "Vary" else False
            if par_varies:
                try:
                    par_min = float(self.field(min))
                    par_max = float(self.field(max))
                except ValueError:
                    QMessageBox.warning(
                        self, "Warning", f"{name} is missing either min or max values."
                    )
                    return False
            else:
                par_min = None
                par_max = None
            try:
                par_value = float(self.field(init) if par_varies else self.field(val))
            except ValueError:  # field probably empty. Choose average as init value
                par_value = (par_min + par_max) / 2
                reply = QMessageBox.question(
                    self,
                    "Warning",
                    f"Init value for {name} is not set.\nSet value to average of lower and upper bound?",
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.setField(init, par_value)
                return False

            self.wizard().model_parameters.add(
                name=name, value=par_value, vary=par_varies, min=par_min, max=par_max
            )
        return True


class Signals(QObject):
    completed = pyqtSignal(dict)


class Worker(QRunnable):
    def __init__(self, fitting_method, args):
        super().__init__()
        self.signal = Signals()
        self.fitting_method = fitting_method
        self.args = args

    @pyqtSlot()
    def run(self):
        self.signal.completed.emit(self.fitting_method(*self.args))


class ResultPage(QWizardPage):
    def __init__(self, parent: QWizard):
        super().__init__(parent)

        self.page_layout = QVBoxLayout(self)

    def initUi(self):
        clearLayout(self.page_layout)
        self.setTitle("Performing Fitting... Please Wait for the Process to Finish")

        self.spinner = WaitingSpinner(self)
        self.page_layout.addWidget(self.spinner)

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
        self.page_layout.addLayout(sub_page_layout)

        self.button = QPushButton("Save Results")
        self.page_layout.addWidget(self.button)
        self.button.clicked.connect(self.save_results)
        self.button.setVisible(False)

    def initializePage(self):
        self.initUi()
        self.perform_fit()
        return super().initializePage()

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
            case "Symbolic":  # "Symbolic"
                self.solver = batch_gen_spectrum_symbolic
            case "Analytical":  # "Analytical":
                self.solver = batch_gen_spectrum_analytical
            case "Numerical":  # "Numerical"
                self.solver = batch_gen_spectrum_numerical

        match self.field("fitting_method"):
            case "Bayesian, MCMC":  # "Bayesian, MCMC"
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
            case "Bayesian, ADVI":  # "Bayesian, ADVI"
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
            case "Nonlinear Least Squares":  # "Nonlinear Least Squares"
                fitting_method = least_squares
                args = (
                    model_parameters,
                    self.model_args,
                    self.data,
                    self.solver,
                    self.field("NLS_algorithm"),
                )

        # Do this in a separate thread!
        pool = QThreadPool.globalInstance()
        self.worker = Worker(fitting_method, args)
        self.worker.signal.completed.connect(self.summarize_fit)
        pool.start(self.worker)
        self.spinner.start()

    def summarize_fit(self, result):
        self.fit_result = result
        self.spinner.stop()
        self.setTitle("Fit Summary")

        match self.field("fitting_method"):
            case "Bayesian, MCMC":  # MCMC
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

                # Diagnose result
                diagnostics = az.summary(self.idata, kind="diagnostics", round_to="none")
                if diagnostics["r_hat"].apply(lambda rhat: rhat > 1.01).any():
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "MCMC chains have not converged.\nReconsider model validity",
                    )
                num_chains = int(self.field("num_chains"))
                num_samples = int(self.field("num_samples"))
                if (
                    diagnostics["ess_tail"]
                    .apply(lambda ess: ess / num_samples < 1 / num_chains)
                    .any()
                    or diagnostics["ess_bulk"]
                    .apply(lambda ess: ess / num_samples < 1 / num_chains)
                    .any()
                ):
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "Effective sample size too low.\nIncrease number of samples.",
                    )

            case "Bayesian, ADVI":  # ADVI
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
            case "Nonlinear Least Squares":  # NLS
                fit_params = result["fit"]
                self.summary.setText(lmfit.fit_report(fit_params, show_correl=False))
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
            if self.field("fitting_method") == "Nonlinear Least Squares"
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
        # fitting_methods = ["Bayesian-MCMC", "Bayesian-ADVI", "Least-Squares"]
        # solvers = ["Symbolic", "Analytical", "Numerical"]
        # saveDir = os.path.join(folderPath, fitting_methods[self.field("fitting_method")], solvers[self.field("solver")])
        saveDir = os.path.join(folderPath, Path(self.wizard().filename).stem)
        try:
            os.makedirs(saveDir, exist_ok=False)
        except FileExistsError:
            reply = QMessageBox.question(self, "Error", "Directory already exists. Overwrite?")
            if reply == QMessageBox.StandardButton.Yes:
                os.makedirs(saveDir, exist_ok=True)
            else:
                return

        self.fig.savefig(os.path.join(saveDir, "best_fit.pdf"), format="pdf")
        with pd.ExcelWriter(os.path.join(saveDir, "best_fit.xlsx"), mode="w") as writer:
            self.wizard().data.to_excel(writer, sheet_name="data", index=False)

            match self.field("fitting_method"):
                case "Bayesian, MCMC":  # MCMC
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
                    ).flatten()[0].get_figure().savefig(
                        os.path.join(saveDir, "pair_plot.pdf"), format="pdf"
                    )
                    az.plot_ess(self.idata, var_names=["~sigma"], relative=True).flatten()[
                        0
                    ].get_figure().savefig(os.path.join(saveDir, "ess_plot.pdf"), format="pdf")
                    with open(os.path.join(saveDir, "fit_summary.txt"), "w") as file:
                        file.write(f"Fitting method: {self.field("fitting_method")}\n")
                        file.write(f"Solver: {self.field("solver")}\n")
                        file.write("\n[Fixed Variables]\n")
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
                case "Bayesian, ADVI":  # ADVI
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
                        file.write(f"Fitting method: {self.field("fitting_method")}\n")
                        file.write(f"Solver: {self.field("solver")}\n")
                        file.write("\n[Fixed Variables]\n")
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
                case "Nonlinear Least Squares":  # NLS
                    df_fit = pd.DataFrame(
                        np.c_[self.offsets, self.fit],
                        columns=["ppm"] + [f"{power:.2g} μT" for power in self.powers],
                    )
                    df_fit.round(3).to_excel(writer, sheet_name="nls", index=False)

                    with open(os.path.join(saveDir, "fit_summary.txt"), "w") as file:
                        file.write(f"Fitting method: {self.field("fitting_method")}\n")
                        file.write(f"Solver: {self.field("solver")}\n")
                        file.write(lmfit.fit_report(self.fit_result["fit"], show_correl=False))
        QMessageBox.information(self, "Info", "Results saved successfully!")


class WizardApp(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Bloch-McConnellizer")
        self.setMinimumSize(QSize(940, 640))

        self.dataPage = DataPage()
        self.modelPage = ModelPage(self)
        self.resultPage = ResultPage(self)

        self.addPage(self.dataPage)
        self.addPage(self.modelPage)
        self.addPage(self.resultPage)


def clearLayout(layout):
    if layout is not None:
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                clearLayout(child.layout())


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
