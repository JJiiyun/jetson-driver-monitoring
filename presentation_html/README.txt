실행 방법
=========

1. 이 폴더의 index.html을 Chrome/Chromium에서 엽니다.
2. 영상 보안 정책 또는 한글 경로 문제를 피하려면 /srv/samba에서 서버를 실행합니다.

   cd /srv/samba
   python3 -m http.server 8000

3. 브라우저에서 다음 주소로 접속합니다.

   http://localhost:8000/drowsiness_presentation_html/

조작법
======

- 오른쪽 방향키 / Space / Enter: 다음
- 왼쪽 방향키 / Backspace: 이전
- F: 전체화면
- O: 슬라이드 전체보기
- N: 발표자 노트
- Home / End: 첫 장 / 마지막 장
- 모바일: 좌우 스와이프

주의
====

- 시연 영상은 상위 폴더의 /srv/samba/시연영상.mp4를 참조합니다.
- 폴더와 시연영상.mp4의 상대 위치를 바꾸면 영상 경로도 수정해야 합니다.
