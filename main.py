import sys
import webbrowser

import psycopg2
from PIL import ImageQt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from ui_loginwindow import *
from ui_mainwindow import *
from img_reconstruction import strip2clean
from img_recognition import image2class


user_now = ''


class LoginWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_LoginWindow()
        self.ui.setupUi(self)

        # 隐藏原框
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # 阴影效果
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(5, 5)
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QtCore.Qt.gray)
        self.ui.frame.setGraphicsEffect(self.shadow)

        # 页面切换
        self.ui.pushButton_Login.clicked.connect(self.go_login)
        self.ui.pushButton_Register.clicked.connect(self.go_register)

        # 登录
        self.ui.pushButton_L_sure.clicked.connect(self.log_in)
        # 注册
        self.ui.pushButton_R_sure.clicked.connect(self.register_in)

        self.show()

    def go_login(self):
        self.ui.stackedWidget_2.setCurrentIndex(0)
        self.ui.stackedWidget.setCurrentIndex(0)
        self.clear_window()

    def log_in(self):
        account = self.ui.lineEdit_L_account.text()
        password = self.ui.lineEdit_L_password.text()
        account_flag = False
        password_flag = False

        # 连接数据库
        conn = psycopg2.connect(database='DataMy', user='postgres', password='123', host='127.0.0.1', port='5432')
        cur = conn.cursor()
        cur.execute("SELECT * FROM users")
        rows = cur.fetchall()
        for row in rows:
            if account == row[0]:
                account_flag = True
                if password == row[1]:
                    password_flag = True
        # 提交修改和关闭连接
        conn.commit()
        conn.close()

        if account_flag and password_flag:
            global user_now
            user_now = account
            self.win = MainWindow()
            self.close()
        elif len(account) == 0 or len(password) == 0:
            self.ui.stackedWidget.setCurrentIndex(1)
        elif not account_flag:
            self.ui.stackedWidget.setCurrentIndex(2)
        elif account_flag and not password_flag:
            self.ui.stackedWidget.setCurrentIndex(3)
        else:
            print("error")

    def go_register(self):
        self.ui.stackedWidget_2.setCurrentIndex(1)
        self.ui.stackedWidget.setCurrentIndex(0)
        self.clear_window()

    def register_in(self):
        account = self.ui.lineEdit_R_account.text()
        pwd_1 = self.ui.lineEdit_R_password_1.text()
        pwd_2 = self.ui.lineEdit_R_password_2.text()

        # 用户名不得重复
        user_flag = False
        # 连接数据库
        conn = psycopg2.connect(database='DataMy', user='postgres', password='123', host='127.0.0.1', port='5432')
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM users WHERE accounts='{account}'")
        rows = cur.fetchall()
        if len(rows) == 0:
            user_flag = True
        # 提交修改和关闭连接
        conn.commit()
        conn.close()

        if len(account) == 0 or len(pwd_1) == 0 or len(pwd_2) == 0:
            self.ui.stackedWidget.setCurrentIndex(1)
        elif pwd_1 != pwd_2:
            self.ui.stackedWidget.setCurrentIndex(4)
        elif not user_flag:
            self.ui.stackedWidget.setCurrentIndex(5)
        else:
            # 插入数据
            # 连接数据库
            conn = psycopg2.connect(database='DataMy', user='postgres', password='123', host='127.0.0.1', port='5432')
            cur = conn.cursor()
            cur.execute(f"INSERT INTO users VALUES('{account}','{pwd_1}')")
            # 提交修改和关闭连接
            conn.commit()
            conn.close()
            # 成功提示
            self.ui.stackedWidget.setCurrentIndex(6)

    def clear_window(self):
        self.ui.lineEdit_L_account.clear()
        self.ui.lineEdit_L_password.clear()
        self.ui.lineEdit_R_account.clear()
        self.ui.lineEdit_R_password_1.clear()
        self.ui.lineEdit_R_password_2.clear()
        self.ui.stackedWidget.setCurrentIndex(0)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 隐藏原框
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # 阴影效果
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(5, 5)
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QtCore.Qt.gray)
        self.ui.frame.setGraphicsEffect(self.shadow)

        # 右上角退出登录
        self.ui.pushButton_logout.clicked.connect(self.log_out)

        # 页面切换
        self.ui.pushButton_home.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_info.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_reco.clicked.connect(self.go_recognition)  #
        self.ui.pushButton_other.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))  #
        self.ui.pushButton_fqa.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(4))
        self.ui.pushButton_my.clicked.connect(self.go_setting)  #

        # 手势识别
        self.ui.pushButton_R_sure.clicked.connect(self.gesture_recognition)

        # 可用扩展
        # 点击按钮打开链接（未完成）
        self.ui.pushButton_O_dataset.clicked.connect(lambda: webbrowser.open("notApply.html"))
        self.ui.pushButton_O_model.clicked.connect(lambda: webbrowser.open("notApply.html"))
        self.ui.pushButton_O_ros.clicked.connect(lambda: webbrowser.open("notApply.html"))
        self.ui.pushButton_O_doc.clicked.connect(lambda: webbrowser.open("notApply.html"))

        # 用户设置
        # 确认修改密码
        self.ui.pushButton_M_sure.clicked.connect(self.change_password)

        self.show()

    def log_out(self):
        global user_now
        self.close()
        self.login = LoginWindow()
        user_now = ''

    def go_recognition(self):
        self.ui.stackedWidget.setCurrentIndex(2)
        self.ui.stackedWidget_3.setCurrentIndex(0)
        self.ui.label_R_raw.clear()
        self.ui.label_R_new.clear()

    def gesture_recognition(self):
        # 清空图像显示区域
        self.ui.label_R_raw.clear()
        self.ui.label_R_new.clear()
        # 清空结果显示区域
        self.ui.stackedWidget_3.setCurrentIndex(0)
        # 获取图像路径
        img_path, img_type = QFileDialog.getOpenFileName(self, "打开图片", "../", "*.jpg;;*.png;;All Files(*)")

        if img_path == '':
            pass  # 防止关闭或取消导入关闭所有页面
        else:
            # 打开图像并显示
            img_raw = QtGui.QPixmap(img_path).scaled(self.ui.label_R_raw.width(), self.ui.label_R_raw.height())
            self.ui.label_R_raw.setPixmap(img_raw)

            # 图像重构
            img_new_pil = strip2clean(img_path)
            # 手势识别
            img_cls = image2class(img_new_pil)

            # 展示重构后的图像
            img_new = ImageQt.toqpixmap(img_new_pil)
            # img_new = img_new.scaled(self.ui.label_R_new.width(), self.ui.label_R_new.height())
            self.ui.label_R_new.setPixmap(img_new)
            # 展示识别后的结果
            self.ui.stackedWidget_3.setCurrentIndex(img_cls + 1)

    def go_setting(self):
        self.ui.stackedWidget.setCurrentIndex(5)
        self.ui.stackedWidget_2.setCurrentIndex(0)
        self.ui.lineEdit_M_pass_1.clear()
        self.ui.lineEdit_M_pass_2.clear()

    def change_password(self):
        global user_now
        pwd_1 = self.ui.lineEdit_M_pass_1.text()
        pwd_2 = self.ui.lineEdit_M_pass_2.text()

        if len(pwd_1) == 0 or len(pwd_2) == 0:
            self.ui.stackedWidget_2.setCurrentIndex(1)
        elif pwd_1 != pwd_2:
            self.ui.stackedWidget_2.setCurrentIndex(2)
        elif pwd_1 == pwd_2:
            # 修改密码
            # 连接数据库
            conn = psycopg2.connect(database='DataMy', user='postgres', password='123', host='127.0.0.1', port='5432')
            cur = conn.cursor()
            cur.execute(f"UPDATE users SET passwords='{pwd_1}' WHERE accounts='{user_now}'")
            # 提交修改和关闭连接
            conn.commit()
            conn.close()
            # 成功提示
            self.ui.stackedWidget_2.setCurrentIndex(3)
        else:
            print("error")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoginWindow()
    sys.exit(app.exec_())


























