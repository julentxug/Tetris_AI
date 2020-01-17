from board import board_prop,board_peso_pos

class pieza_clase:
	
    x=[0,0]
    y=[0,0]
    giro=[]

    # Asignamos a cada nombre de pieza su informacion: los giros que tiene y las casillas 
    # que ocupa respecto a su casilla central (x=0,y=0)
    def asignar_info(self,pieza):
        switcher={
            'Zh': dict(x=[1,0,-1],y=[0,1,1],giro=['Zv']),
            'Zv': dict(x=[0,1,1],y=[-1,0,1],giro=['Zh']),
            'Sh': dict(x=[-1,0,1],y=[0,1,1],giro=['Sv']),
            'Sv': dict(x=[1,1,0],y=[-1,0,1],giro=['Sh']),
            'Tu': dict(x=[-1,0,1],y=[0,1,0],giro=['Tr','Td','Tl']),
            'Tr': dict(x=[0,1,0],y=[-1,0,1],giro=['Td','Tl','Tu']),
            'Td': dict(x=[0,-1,1],y=[-1,0,0],giro=['Tl','Tu','Tr']),
            'Tl': dict(x=[0,-1,0],y=[-1,0,1],giro=['Tu','Tr','Td']),
            'O': dict(x=[-1,-1,0],y=[0,1,1],giro=[]),
            'Jr': dict(x=[0,0,1],y=[-1,1,1],giro=['Jd','Jl','Ju']),
            'Jd': dict(x=[1,1,-1],y=[-1,0,0],giro=['Jl','Ju','Jr']),
            'Jl': dict(x=[-1,0,0],y=[-1,-1,1],giro=['Ju','Jr','Jd']),
            'Ju': dict(x=[-1,-1,1],y=[0,1,0],giro=['Jr','Jd','Jl']),
            'Ll': dict(x=[0,0,-1],y=[-1,1,1],giro=['Lu','Lr','Ld']),
            'Lu': dict(x=[-1,1,1],y=[0,0,1],giro=['Lr','Ld','Ll']),
            'Lr': dict(x=[1,0,0],y=[-1,-1,1],giro=['Ld','Ll','Lu']),
            'Ld': dict(x=[-1,-1,1],y=[-1,0,0],giro=['Ll','Lu','Lr']),
            'Ih' : dict(x=[-2,-1,1],y=[0,0,0],giro=['Iv']),
            'Iv' : dict(x=[0,0,0],y=[2,1,-1],giro=['Ih']),
        }
        info= switcher.get(pieza,dict(x=[0,0],y=[0,0],giro=[]))
        self.x=info['x']
        self.y=info['y']
        self.giro=info['giro']

    def get_giro(self):
         return self.giro

    def get_x(self):
         return self.x

    # Calculamos la posicion de la casilla central de la pieza

    def pos_0_y(self,board,pos):
        max_y=0
        max_peso=0
        for k in range(0,3):
            peso=board_peso_pos(board,pos+self.x[k])
            if (peso-self.y[k])>=max_peso:
                max_peso=peso-self.y[k]
        peso=board_peso_pos(board,pos)
        if peso>=max_peso:
            max_peso=peso
            max_y=0
          
        return max_peso
  

    # Simulamos la colocacion de la pieza en la posicion indicada devolviendo
    # el estado resultante de esta colocacion

    def colocar_pieza(self,board,i):
        y_0=self.pos_0_y(board,i)
        y_pos=19-y_0
        board[y_pos][i]=1
        for k in range(0,3):
            y_pos=19-(y_0+self.y[k])
            x_pos=i+self.x[k]
            board[y_pos][x_pos]=1
        return board
        

