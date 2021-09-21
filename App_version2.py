import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGraphicsView,
                             QGraphicsScene, QHBoxLayout, QAction, QFileDialog,
                             QMessageBox)
from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QIcon, QPixmap
import cv2 as cv
import numpy as np
import Transformaciones as tr



class ventanaPrincipal(QMainWindow):
    def __init__(self):
        super(ventanaPrincipal, self).__init__()

        self.archvo_imagen = ''

        # Titulo de la ventana
        self.tituloVentana()
        # Creación Barra del Menú
        self.barraMenu()
        # Creación barra de herramientas (la de los íconos)
        self.accionesBasicas()
        # Canvas de la imagen Original
        self.canvasOriginal()
        # Canvas de las transformaciones de la imagen
        self.canvasTransformaciones()
        # Creación widget principal
        self.widgetPrincipal()
        # Maximizar Ventana
        self.maximizarVentana()

    # Método para dar el nombre a la aplicación
    def tituloVentana(self):
        self.setWindowTitle("Programa Patologia")

    # Metodo para crear la barra de herramientas (la que tiene los íconos)
    def accionesBasicas(self):
        acciones = self.addToolBar("File")

        cargar = QAction(QIcon('iconos/imagen.svg'), "Cargar", self)
        cargar.triggered.connect(self.cargarImagen)
        acciones.addAction(cargar)
        acciones.addSeparator()

        guardar = QAction(QIcon('iconos/guardar.svg'), "Guardar", self)
        guardar.triggered.connect(self.guardarLienzo)
        acciones.addAction(guardar)
        acciones.addSeparator()

        limpiar = QAction(QIcon('iconos/escoba.svg'), "Limpiar", self)
        limpiar.triggered.connect(self.limpiarLianzo)
        acciones.addAction(limpiar)
        acciones.addSeparator()

        salir = QAction(QIcon("iconos/salida.svg"), 'Salir', self)
        salir.triggered.connect(self.salidaAplicacion)
        acciones.addAction(salir)
        acciones.addSeparator()

    # Método que permite cargar una imagen con formatos png, jpg y bmp
    def cargarImagen(self):
        self.archvo_imagen, self.extension_archivo = QFileDialog.getOpenFileName(self, "Abrir Archivo",
                                                                                 QDir.currentPath(),
                                                                                 "Archivos de imagen(*.png;*.jpg;*.bmp)")
        if self.archvo_imagen != '':
            self.escena1.clear()
            self.escena2.clear()
            self.Imagen_RGB = cv.cvtColor(cv.imread(str(self.archvo_imagen)),
                                          cv.COLOR_BGR2RGB)
            [self.fil, self.col, self.ch] = np.shape(self.Imagen_RGB)

            self.imagen_original = QPixmap(self.archvo_imagen)
            self.escena1.addPixmap(self.imagen_original)
            self.escena2.addPixmap(self.imagen_original)

    def guardarLienzo(self):
        if self.archvo_imagen != '':
            directorio, formato = QFileDialog.getSaveFileName(self, "Guardar Archivo",
                                                              QDir.currentPath(),
                                                              "PNG(*.png);; "
                                                              "JPG(*.jpg);;"
                                                              "BMP(*.bmp)")
            if directorio == '':
                pass
            else:
                try:
                    cv.imwrite(directorio, cv.cvtColor(self.imagenParaGuardar, cv.COLOR_BGR2RGB))
                except:
                    print(directorio)
        else:
            QMessageBox.warning(None, "Atención",
                                "Recuerde que primero debe cargar la imagen antes de"
                                "guardarla")


    # Método para limpiar los lienzos
    def limpiarLianzo(self):
        self.escena1.clear()
        self.escena2.clear()

    # Método para salir de la aplicación en donde se incluye un mensaje de advertencia
    def salidaAplicacion(self):
        ret = QMessageBox.question(self, self.tr("Aplicacion Patología"),
                                   self.tr("Vas a salir del programa.\n" + \
                                           "Deseas Salir?"),
                                   QMessageBox.Yes | QMessageBox.No)
        if ret == QMessageBox.Yes:
            QApplication.exit()

    # Método que permite crear la barra de menú
    def barraMenu(self):

        balance = QAction('Balance de blancos', self)

        # salir.triggered.connect(self.salidaAplicacion)
        tincion = QAction('Normalización Tinción', self)

        menubar = self.menuBar()
        archivo = menubar.addMenu('Archivo')
        lista_transformaciones = menubar.addMenu('Transformaciones')
        ayuda = menubar.addMenu('Acerca de')

        lista_transformaciones.addAction(balance)
        lista_transformaciones.addAction(tincion)
        balance.triggered.connect(self.balanceBlancos)
        tincion.triggered.connect(self.normalizarTincion)
        #

    # Método para crear el lienzo de la izquierda. En este se mostrará siempre la imagen original
    def canvasOriginal(self):
        self.graphicsView1 = QGraphicsView(self)
        self.escena1 = QGraphicsScene(self)
        self.graphicsView1.setScene(self.escena1)

    # Método para crear el lienzo de la derecha. En este se mostrarán las transformaciones
    def canvasTransformaciones(self):
        self.graphicsView2 = QGraphicsView(self)
        self.escena2 = QGraphicsScene(self)
        self.graphicsView2.setScene(self.escena2)
        # Desactivar el scroll de esta ventana
        self.graphicsView2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    # Método que permite posicionar los lienzos en la ventana principal
    def widgetPrincipal(self):
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.campo = QHBoxLayout(self.central)
        self.campo.addWidget(self.graphicsView1)
        self.campo.addWidget(self.graphicsView2)

    # Método que permite maximizar la ventana principal una vez inicie la aplicación
    def maximizarVentana(self):
        QWidget.showMaximized(self)

    # Función que permite
    def balanceBlancos(self):
        if self.archvo_imagen != '':
            self.escena2.clear()
            imagen_BB = tr.white_balance(self.Imagen_RGB)
            self.imagenParaGuardar = np.copy(imagen_BB)
            cv.imwrite('imagen.png', cv.cvtColor(imagen_BB, cv.COLOR_BGR2RGB))
            imagen_BB = QPixmap('imagen.png')
            self.escena2.addPixmap(imagen_BB)
            os.remove('imagen.png')
        else:
            QMessageBox.warning(None, "Atención",
                                "Recuerde que primero debe cargar la imagen antes de"
                                " aplicar alguna transformación")

    def normalizarTincion(self):
        if self.archvo_imagen != '':
            self.escena2.clear()
            imagen_BB = tr.normalizeStaining(self.Imagen_RGB)
            self.imagenParaGuardar = np.copy(imagen_BB)
            cv.imwrite('imagen.png', cv.cvtColor(imagen_BB, cv.COLOR_BGR2RGB))
            imagen_BB = QPixmap('imagen.png')
            self.escena2.addPixmap(imagen_BB)
            os.remove('imagen.png')
        else:
            QMessageBox.warning(None, "Atención",
                                "Recuerde que primero debe cargar la imagen antes de"
                                " aplicar alguna transformación")


aplicacion = QApplication(sys.argv)
ventana = ventanaPrincipal()
ventana.show()
aplicacion.exec_()
