import sys

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QWidget
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget


class plotAndTableWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.create_data()

    def initUI(self):
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.central_widget = QWidget()
        self.vertical_layout = QHBoxLayout()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.vertical_layout)
        self.graphWidget = MatplotlibWidget()

        self.table = pg.TableWidget(editable=True)

        # The 1 after the plot and table essentially tells Qt to expand
        # those widgets evenly so they should take up a similar amount of space.
        self.vertical_layout.addWidget(self.graphWidget, 1)
        self.vertical_layout.addWidget(self.table, 1)

    def create_data(self):
        data = pd.read_excel("example_data.xlsx")
        self.table.setData(data.T.to_dict())
        fig = self.graphWidget.getFigure()
        fig.set_layout_engine("tight")
        ax = fig.add_subplot()
        data.plot(x=0, y=range(1, data.shape[1]), ax=ax, marker="o", lw=0)


def run_program():
    app = QApplication(sys.argv)
    window = plotAndTableWidget()
    window.show()
    app.exec()
    # If you want to close your python interpreter you will need to add
    # sys.exit(app.exec_()) instead of app.exec().


if __name__ == "__main__":
    run_program()
