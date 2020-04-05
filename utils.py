import requests

with open("token.txt") as f:
   token = f.read().strip()

def send_image(IMG_PATH, res):
    # print(IMG_PATH)
    message = '物体を検知しました\n\n'
    for obj in res:
        class_name, val = obj
        message += class_name + " " + str(val) +"\n"
    payload = {'message': message}  # 送信メッセージ
    url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': 'Bearer ' + token}

    files = {'imageFile': open(IMG_PATH, 'rb')}
    r = requests.post(url, headers=headers, params=payload, files=files)  # LINE NotifyへPOST
    # print(r.text)
