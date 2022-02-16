import socket
from functools import lru_cache
from urllib.parse import parse_qs, urlparse, unquote
from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as gbm
from data1_1 import *


def check_dict(type, dict):
    for key, numb in dict.items():
        if key == type:
            return numb


MAX_LINE = 64*1024
MAX_HEADERS = 100


def ml_vehicles():
    df = pd.read_csv('vehicles_data_v_2.csv', index_col=0)
    X = df.drop(['price', 'posting_date'], axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)
    lgb_train = gbm.Dataset(x_train, y_train)
    lgb_eval = gbm.Dataset(x_test, y_test)
    params = {'metric': 'rmse'}
    machine = gbm.train(params,
                        lgb_train,
                        valid_sets=lgb_eval,
                        num_boost_round=3000,
                        early_stopping_rounds=100,
                        verbose_eval=100)
    return machine


def ml_real_estate():
    df = pd.read_csv('ML_real_estate_prepared_v_5.csv', index_col=0)
    X = df.drop(['price', 'date'], axis=1)
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)
    lgb_train = gbm.Dataset(x_train, y_train)
    lgb_eval = gbm.Dataset(x_test, y_test)
    params = {'metric': 'rmse'}
    gbm_machine = gbm.train(params,
                            lgb_train,
                            valid_sets=lgb_eval,
                            num_boost_round=5000,
                            early_stopping_rounds=100,
                            verbose_eval=100)
    return gbm_machine


class Request:
    def __init__(self, method, target, version, rfile):
        self.method = method
        self.target = target
        self.version = version
        self.rfile = rfile
        self.headers = dict()

    def body(self):
        size = self.headers.get('Content-Length')
        if not size:
            return None
        line = self.rfile.read(int(size.strip('/n/n/r')))
        lst = line.split(b"&")
        body_dict = dict()
        for elem in lst:
            pair = elem.split(b'=')
            if pair[1].decode().isdigit():
                body_dict[pair[0].decode()] = int(pair[1].decode())
            else:
                if b'+' in pair[1]:
                    lst = pair[1].split(b'+')
                    result = b''
                    for elem in lst:
                        result += elem
                        result += b' '
                    result = result.decode()
                    result = result[:-1]
                    body_dict[pair[0].decode()] = unquote(result, 'utf-8')
                else:
                    body_dict[pair[0].decode()] = unquote(pair[1], 'utf-8')
        return body_dict

    @property
    def path(self):
        return self.url.path

    @property
    @lru_cache(maxsize=None)
    def query(self):
        return parse_qs(self.url.query)

    @property
    @lru_cache(maxsize=None)
    def url(self):
        return urlparse(self.target)


class Response:
    def __init__(self, status, reason, headers=None, body=None):
        self.status = status
        self.reason = reason
        self.headers = headers
        self.body = body


class HTTPError(Exception):
    def __init__(self, status, reason, body=None):
        self.status = status
        self.reason = reason
        self.body = body


class MyHTTPServer:
    def __init__(self, host, port, server_name):
        self._host = host
        self._port = port
        self._server_name = server_name

    def serve_forever(self):
        serv_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
            proto=0)
        try:
            serv_sock.bind((self._host, self._port))
            serv_sock.listen()
            global ml_car_machine, ml_re_machine
            ml_car_machine = ml_vehicles()
            ml_re_machine = ml_real_estate()
            while True:
                conn, _ = serv_sock.accept()
                try:
                    self.serve_client(conn)
                except Exception as e:
                    pass
        finally:
            serv_sock.close()

    def serve_client(self, conn):  # главная функция, обрабатывающая запрос и подготавливающая ответ клиенту
        try:
            request = self.parse_request(conn)
            if not request:
                wfile = conn.makefile('wb')
                status_line = f'HTTP/1.1 200 OK\r\n'
                wfile.write(status_line.encode('utf-8'))
                wfile.write(b'\r\n')
                wfile.flush()
                wfile.close()
                raise HTTPError('500', 'Wrong GET request')
            response = self.handle_request(request)
            self.send_response(conn, response)
        except ConnectionResetError:
            conn = None
        except HTTPError as e:
            self.send_error(conn, e)
        if conn:
            conn.close()

    def parse_request(self, conn):  # разбор строки запроса, возвращаем объект Request с аттрибутами метод, таргет, версия и словарь с заголовками (параметр: значение)
        rfile = conn.makefile('rb')
        method, target, ver = self.parse_request_line(rfile)
        if method == 'GET':
            return
        request = Request(method, target, ver, rfile)
        request.headers = self.parse_headers(rfile)
        return request

    def parse_headers(self, rfile):  # обрабатываем только строку с параметрами, записываем словарь, который позже записывается в атрибуты Request
        headers = dict()
        while True:
            line = rfile.readline(MAX_LINE + 1)
            if len(line) > MAX_LINE:
                raise Exception('Header line is too long')
            if line in (b'\r\n', b'\n', b''):
                # завершаем чтение заголовков
                break
            pair = line.decode().split(': ')
            headers[pair[0]] = pair[1]
            if len(headers) > MAX_HEADERS:
                raise Exception('Too many headers')
        return headers

    def parse_request_line(self, rfile):  # обрабатываем только первую строку, разбиваем на метод, таргет и версию
        raw = rfile.readline(MAX_LINE + 1)
        if len(raw) > MAX_LINE:
            raise HTTPError(400, 'Bad request',
                            'Request line is too long')
        req_line = str(raw, 'utf-8')
        words = req_line.split()
        if len(words) != 3:
            raise HTTPError(400, 'Bad request',
                            'Malformed request line')
        method, target, ver = words
        if ver != 'HTTP/1.1':
            raise HTTPError(505, 'HTTP Version Not Supported')
        return method, target, ver

    def car(self, body):
        # # manufacturer = 9
        # # model = 4305
        # # condition = 1
        # # cylinders = 3
        # # fuel = 2
        # # odometer = 13900 / 1.609344
        # # title_status = 1
        # # transmission = 1
        # # drive = 2
        # # size = 2
        # # type_of_vehicle = 4
        p = body
        eco = False
        year = float(p.get('year'))
        mark = check_dict(p.get('manufacturer'), car_mark)
        model = check_dict(p.get('model'), car_models)
        cond = check_dict(p.get('condition'), car_cond)
        cyl = check_dict(p.get('cylinders'), cylinders)
        fuel = check_dict(p.get('fuel'), car_fuel)
        if fuel == 4:
            eco = True
        odometer = float(p.get('odometer')) / 1.609344
        st = check_dict(p.get('title_status'), title_status)
        box = check_dict(p.get('transmission'), transmission)
        dr = check_dict(p.get('drive'), drive)
        size = check_dict(p.get('size'), car_size)
        type = check_dict(p.get('type'), car_type)
        test_vehicle = {'year': [year], 'manufacturer': [mark],
                        'model': [model], 'condition': [cond],
                        'cylinders': [cyl], 'fuel': [fuel], 'odometer': [odometer],
                        'title_status': [st], 'transmission': [box], 'drive': [dr], 'size': [size],
                        'type': [type]}
        prediction = ml_car_machine.predict(pd.DataFrame(test_vehicle))
        return prediction * 72.92, eco

    def realestate(self, body):
        p = body
        geo_lat = float(p.get('geo_lat'))
        geo_lon = float(p.get('geo_lon'))
        region = p.get('region')
        region = check_dict(region, regions)
        level = int(p.get('level'))
        levels = int(p.get('levels'))
        rooms = int(p.get('rooms'))
        area = float(p.get('area'))
        kitchen_area = float(p.get('kitchen_area'))
        obj = int(check_dict(p.get('object_type'), obj_types))
        bld = int(check_dict(p.get('building_type'), bld_types))
        test = {'geo_lat': [geo_lat], 'geo_lon': [geo_lon], 'region': [region],
                'level': [level],
                'levels': [levels], 'rooms': [rooms], 'area': [area],
                'kitchen_area': [kitchen_area],
                'object_type': [obj], 'building_type': [bld]}
        prediction = ml_re_machine.predict(pd.DataFrame(test))
        return prediction * 72.92

    def handle_request(self, req):
        body = req.body()
        price = 0
        eco = False
        if body.get('year'):
            price_list = self.car(body)
            price = price_list[0]
            if price_list[1]:
                eco = True
        elif body.get('geo_lat'):
            price = self.realestate(body)
        headers = [('price', price)]
        body = "<html><head><meta charset = 'UTF-8'><style>body{background: linear-gradient(217deg, #FFF5DC, rgba(255,0,0,0) 70%)," \
               "linear-gradient(127deg, #91E2BB, rgba(0,255,0,0) 50%),linear-gradient(306deg, #c3edfd, " \
               "rgba(0,0,255,0) 95%) no-repeat;}div{width: 35%; padding:5% 3%; text-align:center; background:#e5f3f5; " \
               "margin:auto; margin-top: 10%; border-radius: 15px; border: 2px solid white; " \
               "box-shadow: 0 0 8px 4px #e5f3f5} button{background:white; padding: 3% 3%; " \
               "margin-top: 5%;border: 2px solid rgb(192, 191, 191); border-radius: 15px; " \
               "border: 2px solid white; box-shadow: 0 0 8px 4px grey} i{color: green; margin-top: 7%; margin-right: 4%; " \
               "font-size: 280%; float: left;}</style> <script src='https://kit.fontawesome.com/ef692e9f2e.js' " \
               "crossorigin='anonymous'></script></style></head><body><div>"
        if eco:
            body += '<i class="fas fa-leaf">ECO</i>'
        body += f'Цена при данных параметрах составит {str(int(price)).strip("[]")} рублей.'
        body += "<button onclick=\"window.location.href = " \
                "'https://sunshinever.github.io/Valuation-of-property/Valuation%20of%20collateral%20property.html'\">" \
                "Вернуться на главную страницу</button></div></body></html>"
        response = Response(200, 'OK', headers, body)
        print(f'Successful request. Estimated price is {str(int(price)).strip("[]")} rubles.')
        return response

    def send_response(self, conn, resp):
        wfile = conn.makefile('wb')
        status_line = f'HTTP/1.1 {resp.status} {resp.reason}\r\n'
        wfile.write(status_line.encode('utf-8'))

        if resp.headers:
            for (key, value) in resp.headers:
                header_line = f'{key}: {value}\r\n'
                wfile.write(header_line.encode('utf-8'))

        wfile.write(b'\r\n')

        if resp.body:
            wfile.write(bytes(resp.body, 'utf-8'))

        wfile.flush()
        wfile.close()

    def send_error(self, conn, err):
        try:
            status = err.status
            reason = err.reason
            body = (err.body or err.reason).encode('utf-8')
        except:
            status = 500
            reason = b'Internal Server Error'
            body = b'Internal Server Error'
        resp = Response(status, reason,
                        [('Content-Length', len(body))],
                        body)
        self.send_response(conn, resp)


if __name__ == '__main__':
    serv = MyHTTPServer('127.0.0.1', 10001, 'finodays_http')
    try:
        serv.serve_forever()
    except KeyboardInterrupt:
        pass
