import pymysql
import numpy as np
import pickle
from localenv import envloader as el


class MySQLService(object):
    def __init__(self, host='localhost', user='root', password='Lrs12345', db='mydb'):
        self.connection = pymysql.connect(host, user, password, db,
                                          charset='utf8mb4',
                                          cursorclass=pymysql.cursors.DictCursor)

    def commit(self, sql):
        with self.connection.cursor() as cursor:
            # Create a new record
            cursor.execute(sql)

        # connection is not autocommit by default. So you must commit to save your changes.
        self.connection.commit()

    def fetchone(self, sql):
        with self.connection.cursor() as cursor:
            # Read a single record
            cursor.execute(sql)
            result = cursor.fetchone()
        return result

    def fetchall(self, sql):
        with self.connection.cursor() as cursor:
            # Read a single record
            cursor.execute(sql)
            result = cursor.fetchall()
        return result

    def truncate_table(self, table_name):
        sql = "TRUNCATE TABLE %s;" % table_name
        self.commit(sql)

    def drop_table(self, table_name):
        try:
            sql = "DROP TABLE %s" % table_name
            self.commit(sql)
        except:
            pass

    def create_grasp_table(self, table_name):
        sql = '''
        CREATE TABLE %s (
            id int(11) NOT NULL,
            jawwidth blob NOT NULL,
            fc blob NOT NULL,
            hndmat4 blob NOT NULL,
            PRIMARY KEY (`id`)) 
        ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
        ''' % table_name
        self.commit(sql)

    def insert_grasp(self, table_name, grasp_list):
        for i in range(len(grasp_list)):
            grasp = grasp_list[i]
            jawwidth = str(grasp[0])
            fc = str(list(grasp[1]))
            hndmat4 = str([list(row) for row in grasp[2]])
            sql = "INSERT INTO %s (id, jawwidth, fc, hndmat4) VALUES ('%s','%s','%s','%s');" \
                  % (table_name, str(i), jawwidth, fc, hndmat4)
            self.commit(sql)

    def load_all_grasp(self, table_name):
        sql = "SELECT id, jawwidth, fc, hndmat4 FROM %s;" % table_name
        grasp_list = db_service.fetchall(sql)
        result = {}
        for grasp in grasp_list:
            jawwidth = eval(grasp["jawwidth"])
            fc = np.array(eval(grasp["fc"]))
            hndmat4 = np.array(eval(grasp["hndmat4"]))
            result[grasp["id"]] = [jawwidth, fc, hndmat4]

        return result

    def load_grasp_by_id(self, table_name, id):
        sql = "SELECT jawwidth, fc, hndmat4 FROM %s WHERE id='%s';" % (table_name, id)
        grasp = db_service.fetchone(sql)
        jawwidth = eval(grasp["jawwidth"])
        fc = np.array(eval(grasp["fc"]))
        hndmat4 = np.array(eval(grasp["hndmat4"]))

        return [jawwidth, fc, hndmat4]

    def create_objmat4_table(self, table_name):
        sql = '''
        CREATE TABLE %s (
            id int(11) NOT NULL,
            objmat4 blob NOT NULL,
            PRIMARY KEY (id)) 
        ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
        ''' % table_name
        self.commit(sql)

    def insert_objmat4(self, table_name, objmat4_list):
        for i in range(len(objmat4_list)):
            objmat4 = str([list(row) for row in objmat4_list[i]])
            sql = "INSERT INTO %s (id,objmat4) VALUES ('%s','%s');" \
                  % (table_name, i, objmat4)
            self.commit(sql)

    def load_all_objmat4(self, table_name):
        sql = "SELECT id, objmat4 FROM %s;" % table_name
        objmat4_list = db_service.fetchall(sql)
        result = {}
        for objmat4 in objmat4_list:
            objmat4_value = np.array(eval(objmat4["objmat4"]))
            result[objmat4["id"]] = objmat4_value
        return result

    def load_objmat4_by_id(self, table_name, id):
        sql = "SELECT objmat4 FROM %s WHERE id='%s';" % (table_name, id)
        objmat4 = db_service.fetchone(sql)
        return np.array(eval(objmat4["objmat4"]))

    def create_objmat4ngrasp_table(self, table_name, grasp_id_list, objmat4_id_list):
        col_sql = " BOOLEAN, ".join(["grasp_" + str(v) for v in grasp_id_list]) + " BOOLEAN, "
        sql = '''
        CREATE TABLE %s (
            id int(11) NOT NULL,
            objmat4_id int(11) NOT NULL,
            %s
            PRIMARY KEY (id)) 
        ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
        ''' % (table_name, col_sql)
        self.commit(sql)
        for i in range(len(objmat4_id_list)):
            sql = "INSERT INTO %s (id,objmat4_id) VALUES ('%s','%s');" % (table_name, i, objmat4_id_list[i])
            self.commit(sql)

    def pop_objmat4ngrasp_table(self, table_name, grasp_id, objmat4_id, value):
        sql = "UPDATE %s SET grasp_%s = %s WHERE objmat4_id = '%s'" \
              % (table_name, grasp_id, value, objmat4_id)

        self.commit(sql)


if __name__ == '__main__':
    db_service = MySQLService()
    folder_path = el.root + "/motionscript/egg_circle/"
    grasp_table_name = "tst_grasp"
    objmat4_table_name = "tst_objmat4"
    objmat4ngrasp_table_name = "tst_objmat4ngrasp"

    db_service.drop_table(grasp_table_name)
    db_service.create_grasp_table(grasp_table_name)
    grasp_list = pickle.load(open(folder_path + "paintingobj_pregrasp_list.pkl", "rb"))
    # db_service.truncate_table(grasp_table_name)
    db_service.insert_grasp(grasp_table_name, grasp_list)

    db_service.drop_table(objmat4_table_name)
    db_service.create_objmat4_table(objmat4_table_name)
    objmat4_list = pickle.load(open(folder_path + "objmat4_final_list.pkl", "rb"))
    # db_service.truncate_table(objmat4_table_name)
    db_service.insert_objmat4(objmat4_table_name, objmat4_list)

    objmat4ngrasp_pair_dict = pickle.load(
        open(el.root + "/motionscript/egg_circle/objmat4finalngrasp_pair_dict.pkl", "rb"))
    grasp_id_list = list(db_service.load_all_grasp(grasp_table_name))
    objmat4_id_list = list(db_service.load_all_objmat4(objmat4_table_name))

    db_service.drop_table(objmat4ngrasp_table_name)
    db_service.create_objmat4ngrasp_table(objmat4ngrasp_table_name, grasp_id_list, objmat4_id_list)

    for grasp_id in grasp_id_list:
        for objmat4_id in objmat4_id_list:
            value = objmat4ngrasp_pair_dict[grasp_id][objmat4_id]
            db_service.pop_objmat4ngrasp_table(objmat4ngrasp_table_name, grasp_id, objmat4_id, value)
