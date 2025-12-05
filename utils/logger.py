import sys  #βιβλιοθήκη που μας δίνει πρ΄οσβαση στο stdout

#οριμός κλάσης που ανακατευθύνει τη έξοδο σε αρχείο και στην οθόνη  
class OutputLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout                          #κρατάει την αρχική stdout, π.χ. στην οθόνη 
        self.log = open(filename, "w", encoding="utf-8")    #ανοίγει αρχείο log για εγγραφή σε UTF-8
    
    def write(self, message):
        self.terminal.write(message)    #γράφει το μήνυμα κανονικά στην οθόνη
        self.log.write(message)         #γράφει το μήνυμα στο αρχείο log 
        self.log.flush()                #αναγκάζει άμεση εγγραφή στο δίσκο, όχι σε buffer
    
    def flush(self):
        self.terminal.flush()           #χρησιμοποιείται για να μη μένουν δεδομένα στην οθόνη stdout 
        self.log.flush()                #χρησιμοποιείται για να μη μένουν δεδομένα στο αρχείο 
    
    def close(self):
        self.log.close()                #κλείνει το αρχείο log
