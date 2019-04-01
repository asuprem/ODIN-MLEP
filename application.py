
import os, time

global SERVERTIME     
global CREATETIME 
global RECEIVEDTIME
global INTERNAL_TIMER
global FIRST_TIMER
SERVERTIME = 'handshake/servertime'
CREATETIME = 'handshake/createtime'
RECEIVEDTIME = 'handshake/receivedtime'

INTERNAL_TIMER = 0
FIRST_TIMER = True

# Update the time. Wait if file not created. This is a blocking function
def updateTimer():
    global SERVERTIME     
    global CREATETIME 
    global RECEIVEDTIME
    global INTERNAL_TIMER
    global FIRST_TIMER
    updateFlag = False
    if FIRST_TIMER:
        # First time going through this. we will not wait for ReceivedFile
        with open(SERVERTIME, 'w') as servertime:
            servertime.write(str(INTERNAL_TIMER))
        open(CREATETIME, 'w').close()
        FIRST_TIMER = False
    else:
        while not updateFlag:
            #Check if server has received the time
            if not os.path.exists(RECEIVEDTIME):
                pass
            else:
                os.remove(RECEIVEDTIME)
                with open(SERVERTIME, 'w') as servertime:
                    servertime.write(str(INTERNAL_TIMER))
                open(CREATETIME, 'w').close()
                updateFlag = True


def main():
    global SERVERTIME     
    global CREATETIME 
    global RECEIVEDTIME
    global INTERNAL_TIMER
    global FIRST_TIMER

    # Delete all files
    if os.path.exists(SERVERTIME):
        os.remove(SERVERTIME)
    if os.path.exists(CREATETIME):
        os.remove(CREATETIME)
    if os.path.exists(RECEIVEDTIME):
        os.remove(RECEIVEDTIME)

    while True:
        if INTERNAL_TIMER % 5000 == 0:
            print "Internal Timer: ", INTERNAL_TIMER
        updateTimer()
        #time.sleep(5)
        INTERNAL_TIMER += 1
        




if __name__ == "__main__":
    main()