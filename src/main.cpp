// Inference0.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"


//
//  main.cpp
//  Inference
//
//  Created by 舒 天民 on 9/5/14.
//  Copyright (c) 2014 舒 天民. All rights reserved.
//

#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include "math.h"
#include "Trajs.h"
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include "time.h"

#define sqr(x) ((x) * (x))

using namespace std;

typedef struct TPoint {
    double x, y;
    int t;
}TPoint;

typedef struct TRawTraj {
    vector<TPoint> p;
    int ID, role;
}TRawTraj;

typedef struct TTraj {
    vector<TPoint> p;
    vector<TPoint> tp;
    int ID, role, gID;
    TPoint st, ed;
}TTraj;

typedef struct TGroup {
    vector<TTraj> trajs;
}TGroup;

typedef struct TGroupList {
	vector<TGroup> list;
}TGroupList;

typedef struct TAct {
    vector<TGroup> ag;
    double prob;
    int event;
}TAct;

typedef struct TLine {
    TPoint st, ed;
}TLine;

typedef struct TCex {
    double x[2], sigma_x[2][2], theta, sigma_theta, refRes;
    int S1, S2, R1, R2;
}TCex;

typedef struct THist
{
    double h[MAXBIN];
    int n;
}THist;

typedef  struct TCim{
    THist h;
	//double nMbr;
    int S, sg;
}TCim;

typedef struct TTemp {
    vector<TCex> Cex;
    vector<TCim> Cim;
    double nMbr[4];
    double durations, sigma_durations;
	int templateType;
}TTemp;

typedef struct TEvent {
    vector<TTemp> temp;
    set<pair<int, int>> rules;
    vector<int> roles;
	double nSE, nMbr[5];
}TEvent;

typedef struct TSGList{
	vector<TGroup> subGroups;
}TSGList;

void ReadTrajs(int videoID, TGroup *G, double scale);
void SmoothTrajs(TTraj *traj);
int dsvd(float **a, int m, int n, float *w, float **v);
double dist(TPoint p1, TPoint p2);
TPoint minusP(TPoint p1, TPoint p2);
double norm(TPoint p);
double cosTH(TPoint p1, TPoint p2);
double angleL(TLine l1, TLine l2);
double dist(TPoint p1, TPoint p2);
double dist2segment(TPoint p1, TPoint p2, TPoint p3);
double velocity(TPoint p1, TPoint p2);
double orientation(TPoint p1, TPoint p2);

double similarity(TTraj traj1, TTraj traj2, double t1, double t2);
void similarity2(TTraj traj1, TTraj traj2, int t1, int t2, double *sim, double *weight);
bool getConcurrentTraj(const TTraj *traj1, const TTraj *traj2, TTraj *ret1, TTraj *ret2, int *t1, int *t2);
bool getLine(const TTraj *traj, TLine *line);
void getConcurrentGroup(const TGroup *g, TGroupList *conCurrentGroups, int t1, int t2);
void RescaleTrajs(TGroup *g, double scale);
int status(const TTraj *traj);

double max(double a, double b);
double min(double a, double b);

void getGroup(const TAct *act, TSGList *g, int *t1, int *t2, int *t3);
bool truncateGroup(const TGroup *g, TGroup *truncatedGroup, int *t1, int *t2, int startT);
void addGroup(TGroup ag);
void deleteGroup(int k);
void insertAG(TAct *act, TGroup ag);
void deleteAG(TAct *act, int k);
void mergeGroup(int g1, int g2);
void showGroup();

double inter_dist(TAct *g1, TAct *g2);
double inter_group(TAct *g1, TAct *g2);
int cnt_edges(const TAct *g);
int expected_edges(const TAct *g);
int intra(const TAct *g1, const TAct *g2);
void FindCandidateMerge();
void clustering();

int GetBin(double x, double MAX, int bins);
THist CalcHistD(const TGroup *subGroup, int t1, int t2, int bins);
THist CalcHistV(const TGroup *subGroup, int t1, int t2, int bins);
THist CalcHistO(const TGroup *subGroup, int t1, int t2, int bins, double refO);
TTraj CalcCentralTraj(const TGroup *subGroup, int t1, int t2);
THist CalcHistStatic(const TTraj *traj, int t1, int t2, int bins, double beta);
THist CalcHistDynamic(const TGroup *subGroup, int bins);
double HDist(THist h1, THist h2);
double CovHist(const THist *h1, const THist *h2);
THist AddHist(THist h1, THist h2);
THist CalcHistStatic(THist h1, const TTraj *traj, int t1, int t2);
THist Division(THist h, double x);
THist AssignHist(THist h, double x);
THist NormalizeHist(THist h);

double Poisson(int k, double lambda);
double P_Gaussian(double x, double mu, double sigma);
double P_LogNormal(double x, double mu, double sigma);
double Rex(const TTraj *traj1, const TTraj *traj2, int *t1, int *t2, const TCex *Cex, bool *flag);
double CalcResponse(const TSGList *g, int t1, int t2, const TTemp *temp);

map<pair<int, int>, double> sim, w;
map<int, int> IDIndex;

TGroup g, g0;
vector<TAct> act;

vector<int> edge[1000];

TEvent events[20];

int ans[4000];

double Fac[30];
double scale_ref[100];

bool statuses[30][20][2];

vector<int> candidateRole[5000];
vector<pair<int, int>> candidateRoleID;

int maxID;

bool isCar[5000];
bool isObj[5000];

void ReadTemplateConfiguration()
{
	ifstream inf(FILENAME_TEMPCONFIG);
    
	for (int i = 0; i < 20; i++) {
		events[i].temp.clear();
		events[i].rules.clear();
		events[i].roles.clear();
	}
    
	int event, templateType;
	TTemp curTemp;
	TCex curCex;
	TCim curCim;
	int nEx, nIm;
	int nRoles, nRules;
	int curRole;
	pair <int, int> curRule;
	inf >> event;
	while (event != -1) {
		inf >> curTemp.templateType;
		inf >> nEx >> nIm;
        
        curTemp.Cex.clear();
        curTemp.Cim.clear();
        
		for (int i = 0; i < nEx; i++) {
			inf >> curCex.S1 >> curCex.S2 >> curCex.R1 >> curCex.R2;
            curTemp.Cex.push_back(curCex);
		}
        
		events[event].temp.push_back(curTemp);
        
		inf >> event;
	}
    
	inf >> event;
	while (event != -1) {
        
		inf >> nRoles;
		for (int i = 0; i < nRoles; i++) {
			inf >> curRole;
			events[event].roles.push_back(curRole);
		}
        
		inf >> nRules;
		for (int i = 0; i < nRules; i++) {
			inf >> curRule.first >> curRule.second;
			events[event].rules.insert(curRule);
		}
        
		inf >> event;
	}
    
	inf.close();
}

void ReadTemplates()
{
	ifstream inf(FILENAME_TEMPS);
	int nEvents, event, nEx, nIm;
	double a, b, c, d, ss;
    
  //  double refRes;
    
    
    memset(statuses, false, sizeof(statuses));
    
	inf >> nEvents;
	for (int e = 0; e < nEvents; e++) {
		inf >> event;
        
		for (int i = 0; i < events[event].temp.size(); i++) {
			inf >> nEx;
			for (int j = 0; j < nEx; j++) {
                cout << nEvents << " " << event << " " << nEx << " " << events[event].temp[i].Cex.size() << endl;
                
                inf >> events[event].temp[i].Cex[j].S1 >> events[event].temp[i].Cex[j].S2 >> events[event].temp[i].Cex[j].R1 >> events[event].temp[i].Cex[j].R2 >> events[event].temp[i].Cex[j].refRes;
                
                statuses[event][events[event].temp[i].Cex[j].R1][events[event].temp[i].Cex[j].S1] = true;
                statuses[event][events[event].temp[i].Cex[j].R2][events[event].temp[i].Cex[j].S2] = true;
                
                cout << "TEMP " << e << " " << i << " " << j << " " << events[event].temp[i].Cex[j].S1 << " " << events[event].temp[i].Cex[j].S2 << events[event].temp[i].Cex[j].R1 << events[event].temp[i].Cex[j].R2 << events[event].temp[i].Cex[j].refRes << endl;
                
				inf >> events[event].temp[i].Cex[j].x[0] >> events[event].temp[i].Cex[j].x[1] >> a >> b >> c >> d >> events[event].temp[i].Cex[j].theta >> events[event].temp[i].Cex[j].sigma_theta;
				double ss = 1.0 / (a * d - c * b);
				events[event].temp[i].Cex[j].sigma_x[0][0] = ss * d;
				events[event].temp[i].Cex[j].sigma_x[0][1] = ss * (-b);
				events[event].temp[i].Cex[j].sigma_x[1][0] = ss * (-c);
				events[event].temp[i].Cex[j].sigma_x[1][1] = ss * a;
				events[event].temp[i].Cex[j].sigma_theta = 1.0 / events[event].temp[i].Cex[j].sigma_theta;
			}
            
			inf >> nIm;
            TCim Cim;
            Cim.h.n = BINSP;
			for (int j = 0; j < nIm; j++) {
				inf >> Cim.sg >> Cim.S;
				//inf >> events[event].nMbr;
				//inf >> events[event].temp[i].Cim[j].h.n;
				for (int k = 0; k < BINSP; k++) {
					inf >> Cim.h.h[k];
                }
                events[event].temp[i].Cim.push_back(Cim);
			}
            
			int nSubGroups;
			//inf >> nSubGroups;
            nSubGroups = events[event].roles.size();
			for (int j = 0; j < nSubGroups; j++) {
				inf >> events[event].temp[i].nMbr[j];
			}
            
			inf >> events[event].temp[i].durations >> events[event].temp[i].sigma_durations;
            
		}
	}
    
	for (int i = 0; i < events[2].temp[0].Cex.size(); i++)
		cout << events[2].temp[0].Cex[i].R1 << " " <<  events[2].temp[0].Cex[i].R2 << endl;
	system("pause");
    
	inf.close();
}

void init()
{
	Fac[0] = 1.0;
	for (int i = 1; i <= 30; i++)
		Fac[i] = Fac[i - 1] * double(i);
    
	scale_ref[10] = 49.66;
    scale_ref[11] = 56.38;
    scale_ref[12] = 40.29;
    scale_ref[13] = 46.42;
    scale_ref[14] = 50.84;
    scale_ref[15] = 46.04;
    scale_ref[16] = 35.07;
    scale_ref[17] = 51.65;
    scale_ref[18] = 69.77;
    scale_ref[19] = 34.56;
    scale_ref[20] = 39.72;
    scale_ref[56] = 32.45;
    scale_ref[57] = 25.85;
    scale_ref[58] = 22.23;
    scale_ref[59] = 28.16;
    scale_ref[60] = 27.15;
    scale_ref[62] = 23.97;
    scale_ref[63] = 45.73;
    scale_ref[64] = 39.66;
    scale_ref[65] = 50.30;
    scale_ref[68] = 36.62;
    scale_ref[69] = 28.60;
    scale_ref[70] = 36.64;
    scale_ref[71] = 43.90;
    scale_ref[72] = 52.27;
    scale_ref[73] = 37.61;
    scale_ref[74] = 38.45;
    
    ReadTrajs(56, &g, 1.0);
    
	cout << "Complete Trajs Reading.\n" << endl;
    
    for (int i = 0; i < g.trajs.size(); i++)
        SmoothTrajs(&g.trajs[i]);
    
    g0 = g;
    
}

TGroup DPGroup;
TSGList sgList;
TGroupList gList;
double F[1000][10];
int pre[1000][10][2];

TTraj cTraj1, cTraj2;

double DP(TAct *act)
{
	int event = act->event;
	int t1, t2, t3;
    
    TTemp temp1, temp2;
    int minNxtT, maxNxtT;
    int time1, time2;
    
   // int startT = 0;
    
	for (int i = 0; i < act->ag.size(); i++)
		for (int j = 0; j < act->ag[i].trajs.size(); j++) {
			TPoint st = act->ag[i].trajs[j].st, ed = act->ag[i].trajs[j].ed;
			cout << act->ag[i].trajs[j].ID << endl;
			cout << st.t << " " << st.x << " " << st.y << " " << ed.t << " " << ed.x << " " << ed.y << endl;
            
          /*  for (int i2 = 0; i2 < act->ag.size(); i2++)
                for (int j2 = 0; j2 < act->ag[i2].trajs.size(); j2++)
                    if ((i != i2) || (j != j2)) {
                        time1 = 0;
                        time2 = 4000;
                        if (getConcurrentTraj(&(act->ag[i].trajs[j]), &(act->ag[i2].trajs[j2]), &cTraj1, &cTraj2, &time1, &time2)) {
                            int curT = 4000;
                            cout << cTraj1.p.size() << endl;
                            for (int t = 0; t < cTraj1.p.size(); t++)
                                if (dist(cTraj1.p[t], cTraj2.p[t]) < STARTDISTANCE) {
                                    curT = cTraj1.p[t].t;
                                    break;
                                }
                            printf("ID %d ID %d curT %d\n", act->ag[i].trajs[j].ID, act->ag[i2].trajs[j2].ID, curT);
                            startT = max(startT, curT);
                        }
                        
                    }
           */
		}
    
	getGroup(act, &sgList, &t1, &t2, &t3);
    
    bool found = false;
    int startT = 0;
    for (int i = 0; i < events[act->event].temp[0].Cex.size(); i++) {
        int R1 = events[act->event].temp[0].Cex[i].R1 % 10;
        int R2 = events[act->event].temp[0].Cex[i].R2 % 10;
        int S1 = events[act->event].temp[0].Cex[i].S1;
        int S2 = events[act->event].temp[0].Cex[i].S2;
        
        if ((S1 == 1) || (S2 == 1)) {
            
            if ((sgList.subGroups[R1].trajs.size() < 1) && (sgList.subGroups[R2].trajs.size() < 1))
                continue;
            for (int j = 0; j < sgList.subGroups[R1].trajs.size(); j++)
                for (int k = 0; k < sgList.subGroups[R2].trajs.size(); k++) {
                    time1 = t1, time2 = t2;
                    
                  //  cout << "ID " << sgList.subGroups[R1].trajs[j].ID << " " << sgList.subGroups[R1].trajs[j].ID << endl;
                    
                    if (getConcurrentTraj(&(sgList.subGroups[R1].trajs[j]), &(sgList.subGroups[R2].trajs[k]), &cTraj1, &cTraj2, &time1, &time2)) {
                        int curT = 4000;
                        int tt;
                        cout << cTraj1.p.size() << endl;
                        for (int t = 0; t < cTraj1.p.size(); t++)
                            if (dist(cTraj1.p[t], cTraj2.p[t]) < STARTDISTANCE) {
                                curT = cTraj1.p[t].t;
                                tt = t;
                                break;
                            }
                      //  printf("ID %d ID %d curT %d\n", sgList.subGroups[R1].trajs[j].ID, sgList.subGroups[R2].trajs[k].ID, curT);
                        found = true;
                        if (curT < 4000) {
                            if ((tt != 0) || (startT == 0))
                                startT = max(startT, curT);
                        }
                    }
                }

        }
    }
    
    
	cout << "DP...\n";
	cout << "t1 = " << t1 << " t2 = " << t2 << " t3 = " << t3 << endl;
    
    cout << "startT = " << startT << endl;
    
    if (startT >= t2) {
        printf("Too late to start!\n");
    //    return MINLOG - 1.0;
    }
    
    /*    for (int i = 0; i < sgList.subGroups.size(); i++) {
     cout << "SubGroup #" << i << ":\n";
     for (int j = 0; j < sgList.subGroups[i].trajs.size(); j++) {
     cout << sgList.subGroups[i].trajs[j].ID << " " << sgList.subGroups[i].trajs[j].p.size() << " " << sgList.subGroups[i].trajs[j].st.t << " " << sgList.subGroups[i].trajs[j].ed.t << endl;
     cout << sgList.subGroups[i].trajs[j].st.t << " " << sgList.subGroups[i].trajs[j].st.x << " " << sgList.subGroups[i].trajs[j].st.y << " " << sgList.subGroups[i].trajs[j].ed.t << " " << sgList.subGroups[i].trajs[j].ed.x << " " << sgList.subGroups[i].trajs[j].ed.y << endl;
     }
     }
     
     */
    
    memset(pre, -1, sizeof(pre));
    
    double curP;
    
	int totT = t2 - t1;
	int nTemps = events[act->event].temp.size();
	int T = totT / DT, T0 = totT, T2 = (t3 - t1) / DT;
	for (int t = 1; t <= T; t++) {
		for (int j = 0; j < nTemps; j++) {
			F[t][j] = MINLOG - 1.0;
		}
        F[t][BLANKTEMP] = MINLOG - 1.0;
	}
	for (int t = 1; t <= T; t++) {
		for (int tempID = 0; tempID < nTemps; tempID++) {
            time1 = t1, time2 = t1 + GET_TIME(t);
            double time = time2 - time1 + 1;
            temp2 = events[act->event].temp[tempID];
            
            
           /* if ((P_LogNormal(time, temp2.durations, temp2.sigma_durations) < -10.0)) {
                if (temp2.durations < log(time) - EPS) break;
                continue;
            }*/
            
            //printf("init %d %lf %lf %lf \n", temp2.templateType, log(time), temp2.durations, P_LogNormal(log(time), temp2.durations, temp2.sigma_durations));

            
            //getGroup(act, &sgList, &time1, &time2);
			double curP = CalcResponse(&sgList, time1, time2, &(events[act->event].temp[tempID]));
            
            curP *= double(t);
         //   cout << "curP: " << curP << endl;
            
            if (curP > F[t][tempID]) {
                //printf("Update:\nBefore: %lf\n", F[t][step]);
                //cout << t << " " << time2 << " " << tempID << " " << curP << endl;
                F[t][tempID] = curP;
            }
		}
	}
    
   // cout << "F[T][0] = " << F[T][0] << endl;
    
    int maxT = 2 * T / 3;
    for (int t = 1; t < maxT; t++) {
        F[t][BLANKTEMP] = - double (t * DT) / 5;
    }
    
    
    for (int t = 1; t <= T; t++) {
        
        int tempID = BLANKTEMP;
        //printf("F[%d][BLANKTEMP] = %lf\n", t, F[t][tempID]);
        if ((F[t][tempID] > MINLOG) && ((t1 + GET_TIME(t)) <= startT)) {
            
            
            
            //temp1 = events[act->event].temp[tempID];
            int nxtID = 0;
            temp2 = events[act->event].temp[nxtID];
            minNxtT = t + 1;
            maxNxtT = T;
            for (int nxtT = minNxtT; nxtT <= maxNxtT; nxtT++) {
                
                
                
                double time = (double) (GET_TIME(nxtT) - GET_TIME(t));
              //  printf("time = %lf, P_G = %lf\n", time, P_Gaussian(time, temp2.durations, temp2.sigma_durations));
                
                if ((t == 40) && (nxtT == 52))
                    debug_response = true;
                else
                    debug_response = false;
                
            /*    if ((P_LogNormal(time, temp2.durations, temp2.sigma_durations) < -10.0)) {
                    if (temp2.durations < time - EPS) break;
                    continue;
                }*/
                time1 = t1 + GET_TIME(t) + 1;
                time2 = t1 + GET_TIME(nxtT);
                
                
                //getGroup(act, &sgList, &time1, &time2);
                double p = CalcResponse(&sgList, time1, time2, &temp2);
                
                p *= double(nxtT - t + 1);
                
             //   if (time2 == 622)
             //   cout << "a " << time1 << " " << time2 << " " << p << endl;
                
              /*  if (p < (MINLOG * 2.0))
                    return MINLOG * 2.0 - 2.0;*/
                if (p < MINLOG) continue;
                curP = F[t][tempID] + p;
                if (curP > F[nxtT][nxtID]) {
                    
                    //cout << time1 << " " << time2 << " " << tempID << " " << nxtID << " " << curP;
                    
                    F[nxtT][nxtID] = curP;
                    pre[nxtT][nxtID][0] = t;
                    pre[nxtT][nxtID][1] = tempID;
                }
            }
        }
        
        for (int tempID = 0; tempID < nTemps; tempID++)
            if (F[t][tempID] > MINLOG) {
                temp1 = events[act->event].temp[tempID];
                for (int nxtID = 0; nxtID < nTemps; nxtID++) {
                    temp2 = events[act->event].temp[nxtID];
                    minNxtT = t + 1;
                    //maxNxtT = min(t + 12, T);
                    maxNxtT = T;
                    if (events[act->event].rules.count(make_pair(temp1.templateType, temp2.templateType))) {
                        for (int nxtT = minNxtT; nxtT <= maxNxtT; nxtT++) {
                            double time = (double) (GET_TIME(nxtT) - GET_TIME(t));
                          /*  if ((P_LogNormal(time, temp2.durations, temp2.sigma_durations) < -10.0)) {
                                if (temp2.durations < log(time) - EPS) break;
                                continue;
                            }*/
                            time1 = t1 + GET_TIME(t) + 1;
                            time2 = t1 + GET_TIME(nxtT);
                            //getGroup(act, &sgList, &time1, &time2);
                            double p = CalcResponse(&sgList, time1, time2, &temp2);
                            
                            p *= double(nxtT - t + 1);
                            
                        //    if ((nxtID == 2) && (time1 == 622))
                        //        cout << "b " << time1 << " " << time2 << " " << p << endl;
                            
                       /*     if (p < (MINLOG * 2.0))
                                return MINLOG * 2.0 - 2.0;*/
                            if (p < MINLOG) continue;
                            curP = F[t][tempID] + p;
                            
                            //if (nxtT = 42) printf("%lf %lf %lf %lf\n", F[t][step], p, cur, F[nxtT][nxtStep]);
                            
                            if (curP > F[nxtT][nxtID]) {
                                //                    printf("%d %d %d %d %lf = %lf + %lf %lf\n", nxtT, nxtStep, t, step, cur, F[t][step], p,
                                //                        P_Exp((double) (GET_TIME(nxtT) - GET_TIME(t) + 1.0), template[nxtType].duration));
                                //         if (nxtT == T) printf("#2 %d %d %d %d %d %d %lf %lf\n", GET_TIME(t), ref1, ref2, nxtT, nxtR1, nxtR2, cur, p);
                                
                                //cout << time1 << " " << time2 << " " << tempID << " " << nxtID << " " << curP;
                                
                                F[nxtT][nxtID] = curP;
                                pre[nxtT][nxtID][0] = t;
                                pre[nxtT][nxtID][1] = tempID;
                            }
                        }
                    }
                }
            }
    }
    
    double best = F[T][0] / double(T);
    int tempID = 0, t = T;

    for (int endT = T; endT >= T2; endT--) {
        for (int i = 0; i < nTemps; i++) {
            double curBest = F[endT][i] / double(endT);
            if (curBest > best + EPS) {
                best = curBest;
                tempID = i;
                t = endT;
                //cout << best << " " << i << endl;
                
            }
           // cout << endT << " " << i << " " << curBest << endl;
        }
    }
    
    
    int a, b, curTempID;
    int cnt_steps = 0;
    
    cout << "t1 = " << t1 << endl;
    
    while (t > -1) {
        curTempID = tempID;
        //  ref1 = ID[t][templateType][0];
        //  ref2 = ID[t][templateType][1];
        a = max(GET_TIME(pre[t][curTempID][0]) + 1, 1);
        if (a > 1)  a++;
        b = GET_TIME(t) + 1;
        for (int i = a; i <= b; i++)
            ans[i] = curTempID;
        printf("%d %d: TYPE = %d %lf %lf\n", a, b, curTempID, F[t][curTempID], F[t][curTempID] / double(t - pre[t][curTempID][0] + 1));
     //   printf("%d %d: TYPE = %d %lf\n", a, b, events[act->event].temp[curTempID].templateType, F[t][curTempID]);
        if (curTempID != BLANKTEMP)
            cnt_steps++;
        //  getchar();
        tempID = pre[t][curTempID][1];
        t = pre[t][curTempID][0];
    }
    
    //    printf("Poisson: %lf\n", log(Poisson(cnt_steps, events[eventID][part].lambda)));
    
    //    printf("%d %lf\n", cnt_steps, events[eventID][part].lambda);
    
    //    printf("T1 = %d\n", t1);
    
    //  printf("F[3][0] = %lf, F[3][1] = %lf, F[3][2] = %lf t = %d\n", F[3][0], F[3][1], F[3][2], GET_TIME(3) + t1);
    
    /*if (eventID == 1)
     return best + log(Poisson((double) cnt_steps, events[eventID][part].lambda * (double) totT / (300.0)));
     else*/
    
  /*  debug_response = true;
    
    cout << CalcResponse(&sgList, t1 + GET_TIME(40), t1 + GET_TIME(52), &(events[act->event].temp[0])) << endl;
    
    debug_response = false;
  */
    
    act->prob = best;
    return best;
    //return (best / double(T)) /*+ log(Poisson(cnt_steps, 3.0))*/;
}

int statusGroup(const TGroup *g)
{
    double aveV = 0;
    for (int i = 0; i < g->trajs.size(); i++)
        aveV += velocity(g->trajs[i].st, g->trajs[i].ed);
    aveV /= double(g->trajs.size());
    if (aveV < MINV)
        return 0;
    else
        return 1;
}

void showAct(TAct *act)
{
    for (int i = 0; i < act->ag.size(); i++)
        for (int j = 0; j < act->ag[i].trajs.size(); j++) {
            printf("ID #%d Role %d\n", act->ag[i].trajs[j].ID, act->ag[i].trajs[j].role);
        }
    printf("prob = %lf\n", act->prob);
}

void update(TAct *act, TAct *newAct)
{
    *act = *newAct;
    for (int i = 0; i < act->ag.size(); i++)
        for (int j = 0; j < act->ag[i].trajs.size(); j++)
            if (isCar[act->ag[i].trajs[j].ID])
                act->ag[i].trajs[j].role = -10;
            else
                if (isObj[act->ag[i].trajs[j].ID])
                    act->ag[i].trajs[j].role = -50;
}

TAct tmpAct, backupAct;
TTraj staticObj, otherObj;
TCex curTemp;
void InferRole(TAct *act)
{
    candidateRoleID.clear();
    tmpAct = *act;
    for (int i = 0; i < tmpAct.ag.size(); i++) {
        for (int j = 0; j < tmpAct.ag[i].trajs.size(); j++) {
            if ((act->ag[i].trajs[j].role != -50) && (act->ag[i]).trajs[j].role != -10) {
                tmpAct.ag[i].trajs[j].role = -100;
            }
            else {
                staticObj = tmpAct.ag[i].trajs[j];
                staticObj.role = 00;
                tmpAct.ag[i].trajs[j].role = 00;
            }
        }
    }
    
    cout << staticObj.role << endl;
    
    int maxJumps = int(TMP0_ROLE / YITA_ROLE);
    
    for (int i = 0; i < tmpAct.ag.size(); i++) {
        for (int j = 0; j < tmpAct.ag[i].trajs.size(); j++) {
            if (tmpAct.ag[i].trajs[j].role == -100) {
                
                
                int S = status(&(tmpAct.ag[i].trajs[j]));
                
                otherObj = tmpAct.ag[i].trajs[j];
                
                cout << otherObj.ID << " S = " << S << endl;
                
                double best = MINLOG - 1.0;
                int bestR = -1;
               // int best_candidate = 0;
                
                for (int k = 0; k < events[act->event].temp.size(); k++) {
                    for (int e = 0; e < events[act->event].temp[k].Cex.size(); e++) {
                        curTemp = events[act->event].temp[k].Cex[e];
                        //cout << curTemp.R1 << " " << curTemp.R2 << " " << curTemp.S1 << " " << curTemp.S2 << endl;
                        if ((curTemp.R1 == staticObj.role) && (curTemp.S1 == 0) && (curTemp.S2 == S)) {
                            
                            if (S == 0) {
                                bool found = false;
                                for (int r = 0; r < candidateRole[otherObj.ID].size(); r++)
                                    if (candidateRole[otherObj.ID][r] == curTemp.R2) {
                                        found = true;
                                    }
                                if (!found)
                                    candidateRole[otherObj.ID].push_back(curTemp.R2);
                            }
                            
                            otherObj.role = curTemp.R2;
                            int time1 = 0, time2 = 4000;
                            bool flag;
                            double cur = Rex(&(staticObj), &(otherObj), &time1, &time2, &curTemp, &flag);
                            
                            cout << "Try " << otherObj.ID << " " << curTemp.R2 << " " << cur << endl;
                            
                            if (cur > best + EPS) {
                                best = cur;
                                bestR = curTemp.R2;
                              //  best_candidate = int(candidateRole[otherObj.ID].size()) - 1;
                            }
                        }
                    }
                }
                
                tmpAct.ag[i].trajs[j].role = bestR;
                
                if (candidateRole[otherObj.ID].size() > 1) {
                    candidateRoleID.push_back(make_pair(i, j));
                }
                
               // candidateRole[otherObj.ID].erase(candidateRole[otherObj.ID].begin() + best_candidate);
            }
        }
    }
    
    for (int i = 0; i < tmpAct.ag.size(); i++)
        for (int j = 0; j < tmpAct.ag[i].trajs.size(); j++)
            printf("#%d role = %d\n", tmpAct.ag[i].trajs[j].ID, tmpAct.ag[i].trajs[j].role);
    
    double bestP = DP(&(tmpAct)), curP, alpha, u;
    
    cout << "bestP = " << bestP << endl;
    
    
    if (candidateRoleID.size() > 0) {
        int jumpNo = 0;
        for (int attemptNo = 0; attemptNo < 100; attemptNo++) {
            if (jumpNo >= maxJumps) break;
            int candidateID = rand() % (int(candidateRoleID.size()));
            int i = candidateRoleID[candidateID].first, j = candidateRoleID[candidateID].second;
            
            printf("i = %d j = %d\n", i, j);
            
            int roleID = rand() % (int(candidateRole[tmpAct.ag[i].trajs[j].ID].size()) - 1);
            int cnt = -1;
            int k;
            for (k = 0; k < candidateRole[tmpAct.ag[i].trajs[j].ID].size(); k++) {
                if (candidateRole[tmpAct.ag[i].trajs[j].ID][k] != tmpAct.ag[i].trajs[j].role)
                    cnt++;
                if (cnt == roleID) break;
            }
            
            printf("size = %d, cnt = %d, k = %d, roleID = %d\n", int(candidateRole[tmpAct.ag[i].trajs[j].ID].size()), cnt, k, roleID);
            
            backupAct = tmpAct;
            printf("ROLE attempts #%d, ID #%d,  %d -> %d\n", attemptNo, tmpAct.ag[i].trajs[j].ID, tmpAct.ag[i].trajs[j].role, candidateRole[tmpAct.ag[i].trajs[j].ID][k]);
            
            tmpAct.ag[i].trajs[j].role = candidateRole[tmpAct.ag[i].trajs[j].ID][k];
            
            curP = DP(&(tmpAct));
            if (curP > bestP + EPS) {
                alpha = 1.0;
            }
            else {
                alpha = min(exp((curP - bestP) / (TMP0_ROLE - jumpNo * YITA_ROLE)), 1.0);
            }
            double u = double(rand()) / double(RAND_MAX);
            
            printf("ROLE alpha = %lf, u = %lf %lf %lf\n", alpha, u, bestP, curP);
            
            if (u < alpha - EPS) {
                printf("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n");
                printf("Jump #%d: ID #%d, %d -> %d %lf -> %lf\n", ++jumpNo, tmpAct.ag[i].trajs[j].ID, backupAct.ag[i].trajs[j].role, tmpAct.ag[i].trajs[j].role, bestP, curP);
                printf("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n");
                bestP = curP;
                
                showAct(&tmpAct);
            }
            else {
                tmpAct = backupAct;
            }
            printf("RRRRRRRRRRRRRRRR\n");
        }
    }
    
    if (tmpAct.prob > act->prob + EPS) {
        update(act, &(tmpAct));
    }
    
    showAct(act);
}

void detection(int event, int startT, int endT)
{
    int totT = endT - startT;
    int T = totT / DT;
    
    for (int length = 5; length <= T; length += 5) {
        for (int t = 1; t <= T - length; t++) {
            int t1 = t, t2 = t + length - 1;
            if (!truncateGroup(&g0, &g, &t1, &t2, startT)) continue;
            
            sim.clear();
            w.clear();
            for (int i = 0; i < g.trajs.size(); i++) {
                //  if (v.traj[i].type == 0) continue;
                //for (j = 0; j <= i; j++)
                //  printf("\t");
                printf("%6d", g.trajs[i].ID);
                for (int j = 0; j < g.trajs.size(); j++) {
                    
                    //    if (v.traj[j].type == 0) continue;
                    // printf("%d %d:\n", i, j);
                    //   t1 = (double) t;
                    //   t2 = (double) (t1 + 50);
                    int t1 = startTime;
                    int t2 = endTime;
                    //    printf("ID %d ID %d\n", v.traj[i].ID, v.traj[j].ID);
                    double simVal, wVal;
                    similarity2(g.trajs[i], g.trajs[j], t1, t2, &simVal, &wVal);
                    sim[make_pair(g.trajs[i].ID, g.trajs[j].ID)] = simVal;
                    w[make_pair(g.trajs[i].ID, g.trajs[j].ID)] = wVal;
                    printf(" %4.2lf", sim[make_pair(g.trajs[i].ID, g.trajs[j].ID)]);
                    //    printf("\t%d %d %.2lf\n", i, j, similarity(v.traj[i], v.traj[j]));
                    //        printf("===============================\n");
                }
                printf("\n");
            }
            printf("=======================================================\n");
            
            for (int i = 0; i < g.trajs.size(); i++)
                edge[i].clear();
            
            for (int i = 0; i < g.trajs.size(); i++) {
                for (int j = i + 1; j < g.trajs.size(); j++) {
                    if (sim[make_pair(g.trajs[i].ID, g.trajs[j].ID)] > MINRATIO + EPS) {
                        printf("%d %d\n", i, j);
                        edge[IDIndex[g.trajs[i].ID]].push_back(IDIndex[g.trajs[j].ID]);
                        edge[IDIndex[g.trajs[j].ID]].push_back(IDIndex[g.trajs[i].ID]);
                    }
                }
            }
            
            //Each traj forms a single group act
            act.clear();
            TAct curAct;
            TGroup ag;
            for (int i = 0; i < g.trajs.size(); i++) {
              //  if (i == 28)
                //    printf("found %d\n", int(g.trajs[i].p.size()));
                ag.trajs.clear();
                ag.trajs.push_back(g.trajs[i]);
                curAct.ag.clear();
                curAct.ag.push_back(ag);
                act.push_back(curAct);
            }
            
            for (int i = 0; i < act.size(); i++) {
                printf("act[%d].ag.size() = %d\n", i, int(act[i].ag.size()));
                for (int j = 0; j < act[i].ag.size(); j++)
                    printf("act[%d].ag[%d].trajs.size() = %d\n", i, j, int(act[i].ag[j].trajs.size()));
            }
            
            for (int i = 0; i < maxID; i++) {
                candidateRole[i].clear();
            }

            clustering();
            
            
            
            for
        }
    }
        
}


int main(int argc, const char * argv[])
{
    init();
    //clustering();
    
    srand(time(0));
    
	//for (int i = 0; i < act[])
    /*
     */
	ReadTemplateConfiguration();
    ReadTemplates();
    
	
	showGroup();
    
    act[14].event = 2;
    
	mergeGroup(14, 15);
    mergeGroup(14, 16);
    
	for (int i = 0; i < 1; i++)
		for (int j = 0; j < act[14].ag[i].trajs.size(); j++)
			act[14].ag[i].trajs[j].role = 12;
	/*
	for (int i = 1; i < act[14].ag.size(); i++)
		for (int j = 0; j < act[14].ag[i].trajs.size(); j++) {
			if ((act[14].ag[i].trajs[j].ID == 1035) || (act[14].ag[i].trajs[j].ID == 1059) || (act[14].ag[i].trajs[i].ID == 1222)) {
                act[14].ag[i].trajs[j].role = 11;
            }
            else {
                act[14].ag[i].trajs[j].role = 12;
            }
            
        }
    */
    mergeGroup(14, 29);
   /* for (int i = 3; i < act[14].ag.size(); i++)
        for (int j = 0; j < act[14].ag[i].trajs.size(); j++){
            act[14].ag[i].trajs[j].role = 00;
        }
    */
	showGroup();
    
    InferRole(&(act[14]));
 //   cout << DP(&(act[14])) << endl;
    
    
	//showGroup();
    
	system("pause");
    return 0;
}

//******************************************************************************

int status(const TTraj *traj)
{
	double aveVelocity = velocity(traj->st, traj->ed);
    
    if (debug_response)
        cout << "aveVelcoity : " << aveVelocity << endl;
    
	if (aveVelocity > MINDV)
		return 1;
	else
		return 0;
}

TTraj conTraj1, conTraj2;
TLine conLine1, conLine2;
double Rex(const TTraj *traj1, const TTraj *traj2, int *t1, int *t2, const TCex *Cex, bool *flag)
{
	*flag = false;
	if ((traj1->role != Cex->R1) || (traj2->role != Cex->R2))
		return MINRESEX;

    
    if (!getConcurrentTraj(traj1, traj2, &conTraj1, &conTraj2, t1, t2)) {
        
        return MINRESEX;
    }
    
    *flag = true;
    
    if (debug_response) {
		cout << "info:\n";
        cout << traj1->ID << " " << traj2->ID << endl;
        
        cout << traj1->role << " " << traj2->role << " " << status(&conTraj1) << " " << status(&conTraj2) << endl;
        cout << Cex->R1 << " " << Cex->R2 << " " << Cex->S1 << " " << Cex->S2 << endl;
        /*	TPoint st = traj1->st, ed = traj1->ed;
         cout << st.t << " " << st.x << " " << st.y << " " << ed.t << " " << ed.x << " " << ed.y << endl;
         st = traj2->st, ed = traj2->ed;
         cout << st.t << " " << st.x << " " << st.y << " " << ed.t << " " << ed.x << " " << ed.y << endl;
         printf("%lf %lf\n", Cex->theta, Cex->sigma_theta);
         system("pause");
         */
    }

    
    if ((status(&conTraj1) != Cex->S1) || (status(&conTraj2) != Cex->S2))
        return MINRESEX;
        
    double ret;
    
    
    
	if (debug_response)
		cout << "bb\n";
    
    if ((!getLine(&conTraj1, &conLine1) || (!getLine(&conTraj2, &conLine2))))
        return MINRESEX;
    
    double x = fabs(conLine2.ed.x - conLine1.ed.x)- Cex->x[0];
    double y = fabs(conLine2.ed.y - conLine1.ed.y) - Cex->x[1];
    
    ret = -0.5 * ((x * Cex->sigma_x[0][0] + y * Cex->sigma_x[1][0]) * x + (x * Cex->sigma_x[0][1] + y * Cex->sigma_x[1][1]) * y);
    
    //ret += log(0.5 * (sqr(angleL(conLine1, conLine2) - Cex.theta) + 1.0));
    ret += -0.5 * sqr(angleL(conLine1, conLine2) - Cex->theta) * Cex->sigma_theta;
    //ret += angleL
    
    ret -= Cex->refRes;
    
    if (/*debug_res || */debug_response) {
        
		TPoint st = traj1->st, ed = traj1->ed;
		cout << "D " << st.t << " " << st.x << " " << st.y << " " << ed.t << " " << ed.x << " " << ed.y << endl;
		st = traj2->st, ed = traj2->ed;
		cout << "D " << st.t << " " << st.x << " " << st.y << " " << ed.t << " " << ed.x << " " << ed.y << endl;
        
        printf("D %lf %lf %lf %lf\n", conLine2.ed.x - conLine1.ed.x , conLine2.ed.y - conLine1.ed.y, Cex->x[0], Cex->x[1]);
        printf("D %lf %lf %lf\n", angleL(conLine1, conLine2), Cex->theta, Cex->sigma_theta);
        
        printf("ret = %lf + %lf = %lf\n", -0.5 * ((x * Cex->sigma_x[0][0] + y * Cex->sigma_x[1][0]) * x + (x * Cex->sigma_x[0][1] + y * Cex->sigma_x[1][1]) * y), -0.5 * sqr(angleL(conLine1, conLine2) - Cex->theta) * Cex->sigma_theta, ret);
        
		system("pause");
	}
    
    //  printf("(%d %d) ", conLine1.type, conLine2.type);
    //  if ((conLine1.type != Cex.S1) || (conLine2.type != Cex.S2))
    //    ret = -100.0;
    ret = max(ret, -1000.0);
    //ret += -0.5 * sqr(angleL(conLine1, conLine2) - Cex.theta) * Cex.sigma_theta;
    return ret;
}

TGroupList concurrentGroups;
THist curH;
THist aveH;
TTraj tmpP;
double RIm(const TGroup *subGroup, int t1, int t2, const TCim *Cim)
{
    double ret;
    
	double aveVelocity = 0.0;
    for (int i = 0; i < subGroup->trajs.size(); i++)
		aveVelocity += velocity(subGroup->trajs[i].st, subGroup->trajs[i].ed);
	aveVelocity /= double(subGroup->trajs.size());
    
	int status;
	if (aveVelocity < STATICV)
		status = 0;
	else
		status = 1;
    
  //  cout << aveVelocity << " " << " " << status << " " << Cim->S << endl;
    
	if (Cim->S != status)
		return 0.0;
    
	getConcurrentGroup(subGroup, &concurrentGroups, t1, t2);
    
	if (concurrentGroups.list.size() < 1)
		return 0.0;
    
	for (int i = 0; i < concurrentGroups.list.size(); i++) {
		int tt1 = concurrentGroups.list[i].trajs[0].st.t, tt2 = concurrentGroups.list[i].trajs[0].ed.t;
		if (status == 0) {
			tmpP =  CalcCentralTraj(&concurrentGroups.list[i], 0, tt2 - tt1 - 1);
			curH = CalcHistStatic(Cim->h, &tmpP, 0, tt2 - tt1 - 1);
		}
		else curH = CalcHistDynamic(&concurrentGroups.list[i], BINSP);
	}
    
    return CovHist(&curH, &(Cim->h));
}

double Beta(double x, double K)
{
	return exp(K * x) / exp(K);
}

double Poisson(int k, double lambda)
{
	return min(1.0, pow(lambda, (double) k) * exp(-lambda) / Fac[k]);
}

double P_Gaussian(double x, double mu, double sigma)
{
	return -0.5 * sqr(x - mu) / sqr(sigma) /*- log(sqrt(2 * pi) * sigma)*/;
}

double P_LogNormal(double x, double mu, double sigma)
{
    if (mu < EPS) return MINRESEX;
   /* if (debug_response) {
        cout << x << " " << mu << " " << sigma << endl;
        cout << (x * sigma * sqrt(2 * pi)) << " " << exp(-0.5 * (sqr(log(x) - mu)) / sqr(sigma)) << endl;
    }*/
    return log(1.0 / (x * sigma * sqrt(2 * pi)) * exp(-0.5 * (sqr(log(x) - mu)) / sqr(sigma)));
}

double CalcResponse(const TSGList *g, int t1, int t2, const TTemp *temp)
{
    
  /*  for (int i = 0; i < g->subGroups.size(); i++) {
        printf("SG #%d\n", i);
        for (int j = 0; j < g->subGroups[i].trajs.size(); j++)
            printf("ID #%d %d\n", g->subGroups[i].trajs[j].ID, g->subGroups[i].trajs[j].p.size());
    }
    */
	//cout << "T " << t1 << " " << t2 << " " << temp->templateType << endl;
    
	/*if ((t1 == 103) && (t2 == 622) && (temp->templateType == 6))
		debug_response = true;
	else
		debug_response = false;
    */
	int cnt = 0;
	int tot_ex, cnt_ex, best_j;
	double maxR, curR, REX, RIM, ret = 0.0;
	int tt1, tt2;
    
	/*
     for (int i = 0; i < g->subGroups.size(); i++) {
     for (int j = 0; j < g->subGroups[i].trajs.size(); j++) {
     TPoint st = g->subGroups[i].trajs[j].st, ed = g->subGroups[i].trajs[j].ed;
     cout << " " << st.t << " " << st.x << " " << st.y << " " << ed.t << " " << ed.x << " " << ed.y << endl;
     }
     }
     
     system("pause");*/
    
	for (int i = 0; i < temp->Cex.size(); i++) {
		int R1 = temp->Cex[i].R1 % 10;
		int R2 = temp->Cex[i].R2 % 10;
		int O1 = 1 - temp->Cex[i].R1 / 10;
		int O2 = 1 - temp->Cex[i].R2 / 10;
		int S1 = temp->Cex[i].R1 % 10;
		int S2 = temp->Cex[i].R2 % 10;
        
        if (debug_response)	{
            cout << "TEMP: " <<  temp->Cex[i].R1 << " " <<  temp->Cex[i].R2 << endl;
            cout << g->subGroups[R1].trajs.size() << " " << g->subGroups[R2].trajs.size() << endl;
        }
        
		if ((g->subGroups[R1].trajs.size() < 1) || (g->subGroups[R2].trajs.size() < 1)) {
			continue;
		}
		//cnt++;
        REX = 0.0; tot_ex = cnt_ex = 0;
        
      //  cout << "R1 = " << R1 << " R2 = " << R2 << endl;
        
        //cout << g->subGroups[R1].trajs.size() << " " << g->subGroups[R2].trajs.size() << endl;
        
		for (int v2 = 0; v2 < g->subGroups[R2].trajs.size(); v2++) {
		//	printf("%d #%d %d %d\n", i, g->subGroups[R2].trajs[v2].ID, R1, R2);
			if (O1) {
				best_j = -1;
				maxR = MINRESEX;
                bool exists = false;
				for (int j = 0; j < g->subGroups[R1].trajs.size(); j++) {
					tt1 = t1, tt2 = t2;
                    bool flag;
                    
                   // cout << g->subGroups[R1].trajs[j].ID << " " << g->subGroups[R2].trajs[v2].ID << endl;
                    
					curR = Rex(&(g->subGroups[R1].trajs[j]), &(g->subGroups[R2].trajs[v2]), &tt1, &tt2, &(temp->Cex[i]), &flag);
					if (curR > maxR + EPS) {
						maxR = curR;
						best_j = j;
					}
                    if (flag)
                        exists = true;
				}
                
                if (exists)
                    tot_ex++;
                
				if (debug_response) {
					printf("maxR = %lf REX = %lf\n", maxR, exp(maxR));
				}
                
				if ((best_j != -1) && (maxR > MIN_NORMAL_EX)) {
					cnt_ex++;
					REX = exp(maxR);
				}
			}
			else {
				for (int v1 = 0; v1 < g->subGroups[R1].trajs.size(); v1++) {
					tt1 = t1, tt2 = t2;
					bool exits = false;
					curR = Rex(&(g->subGroups[R1].trajs[v1]), &(g->subGroups[R2].trajs[v2]), &tt1, &tt2, &(temp->Cex[i]), &exits);
					if (curR > MIN_NORMAL_EX) {
						REX += exp(curR);
						cnt_ex++;
					}
                    if (exits)
                        tot_ex++;
				}
			}
            
		}
        if (cnt_ex > EPS) {
            ret += REX / double(cnt_ex) * Beta(double(cnt_ex) / double(tot_ex), BETA1);
            cnt++;
        }
        if (debug_response)
            printf("cnt_ex = %d tot_ex = %d ret = %lf\n", cnt_ex, tot_ex, ret);
	}
    
    if (debug_response)
        printf("ret = %lf cnt = %d\n", ret, cnt);
    
	ret /= double(cnt);
	ret *= Beta(double(cnt) / double(temp->Cex.size()), BETA2);
    
    if (debug_response)
        printf("ret = %lf Cex.size = %lf\n", ret, double(temp->Cex.size()));
    
	cnt = 0;
	RIM = 0.0;
	double length, curLength, nMbr;
	for (int i = 0; i < temp->Cim.size(); i++) {
		if (g->subGroups[temp->Cim[i].sg].trajs.size() > 0) {
			getConcurrentGroup(&(g->subGroups[temp->Cim[i].sg]), &gList, t1, t2);
            
            if (debug_response) {
            printf("gList:\n");
            for (int i = 0; i < gList.list.size(); i++) {
                printf("List %d:\n", i);
                for (int j = 0; j < gList.list[i].trajs.size(); j++) {
                    printf("#%d [%d, %d]\n", gList.list[i].trajs[j].ID, gList.list[i].trajs[j].st.t, gList.list[i].trajs[j].ed.t);
                }
            }
            }
            
			if (gList.list.size() > 0) {
				cnt++;
				length = 0.0; nMbr = 0.0; curR = 0.0;
				for (int j = 0; j < gList.list.size(); j++) {
					curLength = double(gList.list[j].trajs[0].ed.t - gList.list[j].trajs[0].st.t + 1.0);
					length += curLength;
					nMbr += double(gList.list[j].trajs.size()) * curLength;
					curR += RIm(&(gList.list[j]), t1, t2, &(temp->Cim[i])) * curLength;
				}
				curR /= length;
				nMbr /= length;
				curR *= Poisson(nMbr, temp->nMbr[i]);
				RIM += curR;
                
                if (debug_response)
                    printf("curR = %lf, nMbr = %lf, Poi = %lf, RIM = %lf\n", curR, nMbr, Poisson(nMbr, temp->nMbr[i]), RIM);
			}
		}
	}
    
	if (cnt == 0) {
		RIM = 0.0;
	}
	else {
		RIM /= double (cnt);
		RIM *= Beta(double(cnt) / double(temp->Cim.size()), BETA3);
	}
    
	if ( temp->Cim.size() < 1)
        RIM = 1.0;
    
	length = t2 - t1 + 1.0;
    
	if (debug_response) {
        cout << length << " " << temp->durations << " " << temp->sigma_durations << endl;
		cout << log(ret) << " " << log(RIM) << " " << P_LogNormal(length, temp->durations, temp->sigma_durations) << endl;
		system("pause");
	}
    
	return log(ret) + log(RIM) + P_LogNormal(length, temp->durations, temp->sigma_durations);
}

//******************************************************************************

TGroup tmpGroup, tmpSG;
void getGroup(const TAct *act, TSGList *g, int *t1, int *t2, int *t3)
{
	*t1 = 10000;
	*t2 = 0;
    *t3 = 0;
	tmpGroup.trajs.clear();
	g->subGroups.clear();
	for (int i = 0; i < act->ag.size(); i++) {
		for (int j = 0; j < act->ag[i].trajs.size(); j++) {
			tmpGroup.trajs.push_back(act->ag[i].trajs[j]);
            if (act->ag[i].trajs[j].role / 10) {
                *t1 = min(*t1, act->ag[i].trajs[j].st.t);
                *t3 = max(*t3, act->ag[i].trajs[j].ed.t);
            }
            *t2 = max(*t2, act->ag[i].trajs[j].ed.t);
		}
	}
    
	for (int i = 0; i < events[act->event].roles.size(); i++) {
		tmpSG.trajs.clear();
		for (int j = 0; j < tmpGroup.trajs.size(); j++) {
			if (tmpGroup.trajs[j].role == events[act->event].roles[i]) {
				tmpSG.trajs.push_back(tmpGroup.trajs[j]);
			}
		}
		g->subGroups.push_back(tmpSG);
	}
}

void addGroup(TGroup ag)
{
    TAct newAct;
    newAct.prob = -1.0; newAct.event = -1;
    newAct.ag.clear();
    newAct.ag.push_back(ag);
}

void deleteGroup(int k)
{
    act.erase(act.begin() + k);
}

void insertAG(TAct *act, TGroup ag)
{
    act->ag.push_back(ag);
}

void deleteAG(TAct *act, int k)
{
    act->ag.erase(act->ag.begin() + k);
}

void mergeAG(int g1, int g2)
{
    act[g1].ag[0].trajs.insert(act[g1].ag[0].trajs.end(), act[g2].ag[0].trajs.begin(), act[g2].ag[0].trajs.end());
    act.erase(act.begin() + g2);
}

void mergeGroup(int g1, int g2)
{
    act[g1].ag.insert(act[g1].ag.end(), act[g2].ag.begin(), act[g2].ag.end());
    act.erase(act.begin() + g2);
}

void showGroup()
{
    int i, j, k;
    for (i = 0; i < act.size(); i++) {
        printf("Group #%d: Event %d Pr = %lf\n", i, act[i].event, act[i].prob);
		printf("ag.size = %d\n", act[i].ag.size());
        for (j = 0; j < act[i].ag.size(); j++)
            for (k = 0; k < act[i].ag[j].trajs.size(); k++) {
                printf("Traj: %d %d %d\n", act[i].ag[j].trajs[k].ID, int(act[i].ag[j].trajs[k].p.size()), act[i].ag[j].trajs[k].role);
				TPoint st = act[i].ag[j].trajs[k].st;
				TPoint ed = act[i].ag[j].trajs[k].ed;
                
				//cout << st.t << " " << st.x << " " << st.y << " " << ed.t << " " << ed.x << " " << ed.y << endl;
			}
    }
}

//******************************************************************************

double distArray[50];

double inter_dist(TAct *g1, TAct *g2)
{
    int i, j, ii, jj, n;
    double tmp, cnt = 0.0, tot = 0.0;
    for (i = 0; i < g1->ag.size(); i++)
        for (j = 0; j < g1->ag[i].trajs.size(); j++) {
            n = 0;
            for (ii = 0; ii < g2->ag.size(); ii++)
                for (jj = 0; jj < g2->ag[ii].trajs.size(); jj++)
                    distArray[n++] = w[make_pair(g1->ag[i].trajs[j].ID, g2->ag[ii].trajs[jj].ID)];
            for (ii = 0; ii < n; ii++)
                for (jj = ii + 1; jj < n; jj++)
                    if (distArray[ii] > distArray[jj] + EPS) {
                        tmp = distArray[ii]; distArray[ii] = distArray[jj]; distArray[jj] = tmp;
                    }
            if (n % 2) {
                n = n / 2 + 1;
            }
            else {
                n /= 2;
            }
            
            for (ii = 0; ii < n; ii++) {
                cnt += distArray[ii];
                tot += 1.0;
            }
        }
    
    return cnt / tot;
}

double inter_group(TAct *g1, TAct *g2)
{
    return (inter_dist(g1, g2) + inter_dist(g2, g1)) * 0.5;
}

int appeared[10000];

int cnt_edges(const TAct *g)
{
    int i, j;
    int ret = 0;
    memset(appeared, 0, sizeof(appeared));
    for (i = 0; i < g->ag.size(); i++)
        for (j = 0; j < g->ag[i].trajs.size(); j++)
            appeared[IDIndex[g->ag[i].trajs[j].ID]] = 1;
    for (i = 0; i < 10000; i++)
        if (appeared[i]) {
            for (j = 0; j < edge[i].size(); j++)
                if (appeared[edge[i][j]])
                    ret++;
        }
    return ret / 2;
}

int expected_edges(const TAct *g)
{
    int i, ret = 0;
    for (i = 0; i < g->ag.size(); i++)
        ret += g->ag[i].trajs.size();
    
    if (ret % 2)
        return ((ret - 1) / 2 + 1) * (ret - 1) / 2;
    else
        return (ret / 2) * (ret / 2);
}

TAct mg;

int intra(const TAct *g1, const TAct *g2)
{
    int i;
    mg.ag.clear();
    // cout << g1->ag.size() << " " << g2->ag.size() << endl;
    for (i = 0; i < g1->ag.size(); i++)
        mg.ag.push_back(g1->ag[i]);
    for (i = 0; i < g2->ag.size(); i++) {
        //        cout << g2->ag[i].trajs.size() << endl;
        mg.ag.push_back(g2->ag[i]);
    }
    
    return (cnt_edges(&mg) >= (expected_edges(&mg) + cnt_edges(g1) - expected_edges(g1) + cnt_edges(g2) - expected_edges(g2)));
}

vector<pair<int, int>> candidateMerge;
vector<int> wCandidateMerge;

void FindCandidateMergeInit()
{
    int i, j;
    
    pair<int, int> tmp;
    
    double tmp_w;
    
    bool found;
    
    TGroup V1, V2;
    candidateMerge.clear();
    wCandidateMerge.clear();
    for (i = 0; i <  act.size(); i++)
        for (j = i + 1; j < act.size(); j++) {
            //   cout << i << " " << act[i].ag[0].trajs.size() << " " << j << " " << act[j].ag[0].trajs.size() << endl;
            
            found = false;
            for (int k1 = 0; (k1 < act[i].ag.size()) && (!found); k1++) {
                for (int l1 = 0; (l1 < act[i].ag[k1].trajs.size()) && (!found); l1++)
                    for (int k2 = 0; (k2 < act[j].ag.size()) && (!found); k2++)
                        for (int l2 = 0; (l2 < act[j].ag[k2].trajs.size()) && (!found); l2++) {
                            if (act[i].ag[k1].trajs[l1].role != act[j].ag[k2].trajs[l2].role) {
                                found = true;
                                break;
                            }
                        }
            }
            
            if (found) continue;
            
            if (intra(&(act[i]), &(act[j]))) {
                
                if (debug_FindCandidateMergeInit)
                    printf("candidate: %d %d\n", i, j);
                
                candidateMerge.push_back(make_pair(i, j));
                wCandidateMerge.push_back(inter_group(&(act[i]), &(act[j])));
            }
        }
    
    for (i = 0; i < candidateMerge.size(); i++)
        for (j = i + 1; j < candidateMerge.size(); j++)
            if (wCandidateMerge[i] > wCandidateMerge[j] + EPS) {
                tmp_w = wCandidateMerge[i]; wCandidateMerge[i] = wCandidateMerge[j]; wCandidateMerge[j] = tmp_w;
                tmp = candidateMerge[i]; candidateMerge[i] = candidateMerge[j]; candidateMerge[j] = tmp;
            }
}

void clustering()
{
	/*
     for (int i = 0; i < act.size(); i++) {
     printf("act[%d].ag.size() = %d\n", i, act[i].ag.size());
     for (int j = 0; j < act[i].ag.size(); j++)
     printf("act[%d].ag[%d].trajs.size() = %d\n", i, j, act[i].ag[j].trajs.size());
     }
     */
    
    while (1) {
        FindCandidateMergeInit();
        if (candidateMerge.size() < 1) break;
        cout << candidateMerge.size() << endl;
        printf("Merge: %d %d\n", candidateMerge[0].first, candidateMerge[0].second);
        mergeAG(candidateMerge[0].first, candidateMerge[0].second);
    }
    
    /*
    while (true) {
    int kk = -1; bool found = false;
    for (int i = 0; i < act.size(); i++) {
        if (act[i].ag[0].trajs[0].role > -100) {
            if (kk == -1) {
                kk = i;
            }
            else {
                mergeAG(kk, i);
                break;
            }
        }
        else {
            act[k++] = act[i];
        }
    }*/

    
	for (int i = 0; i < act.size(); i++) {
		act[i].event = -1;
		act[i].prob = -2000.0;
	}
    
    showGroup();
}

//******************************************************************************

//Read trajs from video #vidoeID and rescale them
void ReadTrajs(int videoID, TGroup *G, double scale)
{
    char name[100];
    sprintf(name, FILENAME_TRAJS, videoID);
    //cout << name << endl;
	printf("%s\n", name);
	ifstream fp(name);
    //FILE *fp; fopen_s(&fp, name, "r");
    
    int ID, T, n = 0;
    TTraj curTraj;
    TPoint pre, cur;
    int w, h;
    
	int minT = 4000, maxT = 0;
    
	maxID = 0;
    
    G->trajs.clear();
    IDIndex.clear();
    
    // fscanf_s(fp, "%d", &ID);
	fp >> ID;
    while (ID != -1) {
		maxID = max(maxID, ID);
        // fscanf_s(fp, "%d", &T);
		fp >> T;
        cout << ID << " " << T << endl;
        IDIndex[ID] = n++;
        curTraj.p.clear();
        curTraj.ID = ID;
        curTraj.gID = ID;
        curTraj.role = -100;
        for (int i = 0; i < T; i++) {
            //fscanf_s(fp, "%d %lf %lf %d %d", &cur.t, &cur.x, &cur.y, &w, &h);
			fp >> cur.t >> cur.x >> cur.y >> w >> h;
            cur.x += w / 2, cur.y += h / 2;
            cur.x *= scale; cur.y *= scale;
            if (i > 0) {
                if (pre.t < cur.t - 1) {
                    int dt = cur.t - pre.t;
                    double dx = (cur.x - pre.x) / double(dt);
                    double dy = (cur.y - pre.y) / double(dt);
                    for (int j = 1; j < dt; j++) {
                        pre.x += dx;
                        pre.y += dy;
                        pre.t++;
                        curTraj.p.push_back(pre);
                    }
                }
            }
            curTraj.p.push_back(cur);
            pre = cur;
        }
        curTraj.st = curTraj.p[0];
        curTraj.ed = curTraj.p[curTraj.p.size() - 1];
        
		if (curTraj.p.size() > 60) {
            
			minT = min(minT, curTraj.st.t);
			maxT = max(maxT, curTraj.ed.t);
            
			G->trajs.push_back(curTraj);
		}
        //fscanf_s(fp, "%d", &ID);
		fp >> ID;
    }
    
	sprintf(name, FILENAME_TRAJSCAR, videoID);
	fp.close();
	fp.open(name);
    
	TTraj traj;
	
	while (true) {
		fp >> T;
		if (T < 1) break;
        
		traj.p.clear();
		
		traj.role = -10;
		for (int i = 0; i < T; i++) {
			fp >> cur.t >> cur.x >> cur.y;
			traj.p.push_back(cur);
		}
        
		if (T <= 60)
			continue;
		
		traj.st = traj.p[0]; traj.ed = traj.p[traj.p.size() - 1];
		traj.ID = ++maxID;
        
		minT = min(minT, traj.st.t);
		maxT = max(maxT, traj.ed.t);
        
        if (T < 200)
            continue;
        
		G->trajs.push_back(traj);
	}
    
	sprintf(name, FILENAME_TRAJSOBJ, videoID);
	fp.close();
	fp.open(name);
    
	while (true)
	{
		fp >> cur.x;
		if (cur.x < 0) break;
		fp >> cur.y;
		traj.p.clear();
		traj.ID = ++maxID;
		traj.role = -50;
		for (int t = minT; t <= maxT; t++) {
			cur.t = t;
			traj.p.push_back(cur);
		}
		traj.st = traj.p[0]; traj.ed = traj.p[traj.p.size() - 1];
		G->trajs.push_back(traj);
	}
    
    
  //  scale_ref[videoID] = 100.0;
    
	RescaleTrajs(G, 10.0 / scale_ref[videoID]);
    
    printf("Scale = %lf\n", 10.0 / scale_ref[videoID]);
    
	MINDIST *= (10.0 / scale_ref[videoID]);
	MINV *= (10.0 / scale_ref[videoID]);
	NORMD *= (10.0 / scale_ref[videoID]);
	NORMV *= (10.0 / scale_ref[videoID]);
    STATICV *= (10.0 / scale_ref[videoID]);
    
    cout << "TRAJS:\n";
    for (int i = 0; i < G->trajs.size(); i++) {
        printf("#%d %d\n", G->trajs[i].ID, G->trajs[i].p.size());
    }
    
    /* TLine line;
     for (int i = 0; i < G->trajs.size(); i++) {
     getLine(&G->trajs[i], &line);
     }*/
}

//Smooth the trajs with window size of SDT
void SmoothTrajs(TTraj *traj)
{
    int T = 0;
    int size = int(traj->p.size()) - SDT;
    for (int i = 0; i <= size; i += SDT) {
        traj->p[T] = traj->p[i];
        for (int j = 1; j < SDT; j++) {
            traj->p[T].x += traj->p[i + j].x;
            traj->p[T].y += traj->p[i + j].y;
        }
        traj->p[T].x /= double(SDT);
        traj->p[T].y /= double(SDT);
        traj->p[T].t = (traj->p[T].t + SDT / 2) / SDT;
        T++;
    }
    traj->p.resize(T);
	traj->st = traj->p[0];
	traj->ed = traj->p[traj->p.size() - 1];
    
	cout << traj->p.size() << endl;
}

float r0[3];

bool getLine(const TTraj *traj, TLine *line)
{
    float **a, **v, *w;
    int T = int(traj->p.size());
    if (T < 3) return false;
    
    a = (float **) malloc(sizeof(float *) * T);
    for (int j = 0; j < T; j++)
        a[j] = (float*) malloc(sizeof(float) * 3);
    
    for (int j = 0; j < 3; j++)
        r0[j] = 0.0;
    
    int k = 0;
    for (int j = 0; j < T; j++) {
        // printf("add %lf; ", V.traj[i].p[j + k].t);
        a[j][0] = traj->p[j + k].t - traj->p[k].t;
        a[j][1] = traj->p[j + k].x;
        a[j][2] = traj->p[j + k].y;
        r0[0] += a[j][0];
        r0[1] += a[j][1];
        r0[2] += a[j][2];
    }
    //printf("\n");
    
    for (int j = 0; j < 3; j++)
        r0[j] /= double(T);  //mean of t, x, y in a[][]
    
    //  for (j = 0; j < 3; j++)
    //    printf("r0[%d] = %lf\n", j, r0[j]);
    
    for (int j = 0; j < T; j++) {
        a[j][0] -= r0[0];
        a[j][1] -= r0[1];
        a[j][2] -= r0[2];
    }
    
    //    printf("Before: Tmin Tmax %lf %lf\n", a[0][0], a[V.traj[i].T - 1][0]);
    
    w = (float *) malloc(3 * sizeof(float));
    v = (float **) malloc(sizeof(float *) * 3);
    for (int j = 0; j < 3; j++)
        v[j] = (float *) malloc(sizeof(float) * 3);
    int ed = traj->ed.t, st = traj->st.t;
    if (!(dsvd(a, ed - st, 3, w, v)))
        return false;
    
    double Tmin =  (st - st - r0[0]) / v[0][0];
    double Tmax =  (ed - st - r0[0]) / v[0][0];
    
    //     printf("Tmin Tmax %lf %lf\n", a[0][0], a[V.traj[i].T - 1][0]);
    
    line->st.t = int (v[0][0] * Tmin + r0[0]);  //start point
    line->st.x = v[1][0] * Tmin + r0[1];
    line->st.y = v[2][0] * Tmin + r0[2];
    
    line->ed.t = int (v[0][0] * Tmax + r0[0]);  //end point
    line->ed.x = v[1][0] * Tmax + r0[1];
    line->ed.y = v[2][0] * Tmax + r0[2];
    
    if (debug_getLine) {
        cout << traj->ID << " " << T << endl;
        cout << line->st.t << " " << line->st.x << " " << line->st.y << " " << line->ed.t << " " << line->ed.x << " " << line->ed.y << endl;
    }
    
    free(a); free(w); free(v);
    
    return true;
}

//Get the concurrent part of traj1 and traj2 in [t1, t2]
bool getConcurrentTraj(const TTraj *traj1, const TTraj *traj2, TTraj *ret1, TTraj *ret2, int *t1, int *t2)
{
    ret1->p.clear();
    ret2->p.clear();
    ret1->role = traj1->role; ret2->role = traj2->role;
    ret1->ID = traj1->ID; ret2->ID = traj2->ID;
    ret1->gID = traj1->gID; ret2->gID = traj2->gID;
    
    int size1 = int(traj1->p.size());
    while ((traj1->p[size1 - 1].t < 1) && (size1 > 0)) size1--;
    int size2 = int(traj2->p.size());
    while ((traj2->p[size2 - 1].t < 1) && (size2 > 0)) size2--;
    
    *t1 = max(*t1, traj1->p[0].t);
    *t2 = min(*t2, traj1->p[size1 - 1].t);
    *t1 = max(*t1, traj2->p[0].t);
    *t2 = min(*t2, traj2->p[size2 - 1].t);
    //  printf("get T1 = %lf T2 = %lf\n", *t1, *t2);
    if (*t1 < *t2) {
        int i;
        for (i = 0; i < size1; i++) {
            if ((traj1->p[i].t >= *t1) && (traj1->p[i].t <= *t2)) {
                ret1->p.push_back(traj1->p[i]);
            }
        }
        for (i = 0; i < size2; i++) {
            if ((traj2->p[i].t >= *t1) && (traj2->p[i].t <= *t2)) {
                ret2->p.push_back(traj2->p[i]);
            }
        }
        
		ret1->st = ret1->p[0]; ret1->ed = ret1->p[ret1->p.size() - 1];
		ret2->st = ret2->p[0]; ret2->ed = ret2->p[ret2->p.size() - 1];
        return true;
    }
    return false;
}

TTraj simTraj1, simTraj2;
double similarity(TTraj traj1, TTraj traj2, int t1, int t2)
{
    if (!getConcurrentTraj(&traj1, &traj2, &simTraj1, &simTraj2, &t1, &t2))
        return 0.0;
    
    double cnt = 0.0;
    double tot = 0.0;
    int i = 0;
    
    int T = min(int(simTraj1.p.size()), int(simTraj2.p.size()));
    if (T <= 5) return 0.0;
    
    //  printf("%lf %lf\n", t1, t2);
    
    for (i = 5; i < T; i += 5) {
        
        tot += 1.0;
        TPoint v1 = minusP(simTraj1.p[i], simTraj1.p[i - 5]);
        TPoint v2 = minusP(simTraj2.p[i], simTraj2.p[i - 5]);
        double sV1, sV2;
        sV1 = dist(simTraj1.p[i], simTraj2.p[i]);
        sV2 = dist(v1, v2) / 5.0;
        
        /* if ((traj1.type == 0) && (traj1.role != -8) || (traj2.type == 0) && (traj2.role != -8))
         sV2 = 0.0;*/
        
        //   if ((sV1 > MINDIST) || (sV2 > MINV)) continue;
        double simValue = ((sV1 / NORMD) * ALPHA + ((sV2) / NORMV) * (1 - ALPHA));
        if ((traj1.ID == 6) && (traj2.ID == 42)) {
            printf("t1 %d t2 %d\n", simTraj1.p[i].t, simTraj2.p[i].t);
            printf("%lf %lf %lf %lf\n",  v1.x, v1.y, v2.x, v2.y);
            printf("%d %lf %lf %lf\n", i, simValue, sV1, sV2);
        }
        
        //  printf("%d: %lf %lf sim = %lf\n", i, simTraj1.p[i].t, simTraj2.p[i].t, simValue);
        // if ((dist(traj1.p[i], traj2.p[i]) < MINDIST) || ((dist(v1, v2) / 15.0) < MINV))
        // if ((dist(traj1.p[i], traj2.p[i]) < MINDIST) && ((dist(v1, v2) / 15.0) < MINV))
        if (simValue < MINDV - EPS)
            
            cnt += 1.0;
        //    printf("%d: %lf %lf %lf %lf\n", (int) traj1.p[i].t, dist(traj1.p[i], traj2.p[i]),
        //   dist(v1, v2), velocity(traj1.p[i], traj1.p[i - 5]), velocity(traj2.p[i], traj2.p[i - 5]));
    }
    if (tot < EPS) return 0.0;
    //  printf("%lf %lf\n", cnt, tot);
    return cnt / tot;
}

void similarity2(TTraj traj1, TTraj traj2, int t1, int t2, double *sim, double *weight)
{
    if (!(getConcurrentTraj(&traj1, &traj2, &simTraj1, &simTraj2, &t1, &t2))) {
        *sim = 0.0;
        *weight = 10000.0;
        return;
    }
    
    
    *weight = 0.0;
    double cnt = 0.0, tot = 0.0;
    
    int T = int(min(simTraj1.p.size(), simTraj2.p.size()));
    if (T <= 5) {
        *sim = 0.0;
        *weight = 10000.0;
        return;
    }
    
    for (int i = 5; i < T; i += 5) {
        tot += 1.0;
        double sV1, sV2;
        sV2 = 0.0;
        TPoint v1, v2;
        for (int j = 1; j <= 5; j++) {
            v1 = minusP(simTraj1.p[i], simTraj1.p[i - j]);
            v2 = minusP(simTraj2.p[i], simTraj2.p[i - j]);
            sV2 += dist(v1, v2) / double(j);
        }
        
        sV1 = dist(simTraj1.p[i], simTraj2.p[i]);
        sV2 = dist(v1, v2) / 5.0;
        if (((traj1.ID == 590) && (traj2.ID == 611)) || ((traj2.ID == 611) && (traj1.ID == 590))) {
            //cout << simTraj1.p[i] << " " << simTraj1.p[i - 5]
            cout << "Check: " << sV1 << " " << sV2 << " " << cnt << " " << tot << endl;
        }
        if ((sV1 < MINDIST) && (sV2 < MINV))
            cnt += 1.0;
        double simValue = ((sV1 / NORMD) * ALPHA + ((sV2) / NORMV) * (1 - ALPHA));
        
        *weight += simValue;
    }
    
    if (tot < EPS) {
        *sim = 0.0;
        *weight = 10000.0;
    }
    else {
        *sim = cnt / tot;
        if ((cnt * tot) < EPS) {
            *weight = 10000.0;
        }
        else {
            *weight /= (cnt * tot);
        }
    }
    
}

void RescaleTrajs(TGroup *g, double scale)
{
	for (int i = 0; i < g->trajs.size(); i++) {
		for (int j = 0; j < g->trajs[i].p.size(); j++) {
			g->trajs[i].p[j].x *= scale;
			g->trajs[i].p[j].y *= scale;
		}
		g->trajs[i].st = g->trajs[i].p[0];
		g->trajs[i].ed = g->trajs[i].p[g->trajs[i].p.size() - 1];
	}
}

void TruncateTrajs(const TTraj *traj, TTraj *truncatedTraj, int t1, int t2)
{
	*truncatedTraj = *traj;
	truncatedTraj->p.clear();
	for (int i = 0; i < traj->p.size(); i++)
		if (INTIME(traj->p[i].t, t1, t2)) {
			truncatedTraj->p.push_back(traj->p[i]);
		}
	truncatedTraj->st = truncatedTraj->p[0];
	truncatedTraj->ed = truncatedTraj->p[truncatedTraj->p.size() - 1];
}

TTraj truncatedTraj;
TGroup curGroup;
bool added[100];
void getConcurrentGroup(const TGroup *g, TGroupList *conCurrentGroups, int t1, int t2)
{
	memset(added, false, sizeof(added));
	
	int pre = t1;
	bool first;
	conCurrentGroups->list.clear();
	for (int t = t1; t <= t2; t++) {
		first = true;
		for (int i = 0; i < g->trajs.size(); i++) {
			if (INTIME(t, g->trajs[i].st.t, g->trajs[i].ed.t)) {
				if (!added[i]) {
					if ((first) && (t > t1)) {
						curGroup.trajs.clear();
						for (int j = 0; j < g->trajs.size(); j++) {
							if (added[j]) {
								TruncateTrajs(&g->trajs[j], &truncatedTraj, pre, t - 1);
								curGroup.trajs.push_back(truncatedTraj);
							}
						}
						if (curGroup.trajs.size())
							conCurrentGroups->list.push_back(curGroup);
						pre = t;
					}
                    
					added[i] = true;
					first = false;
				}
			}
			else {
				if (added[i]) {
					if ((first) && (t > t1)) {
						curGroup.trajs.clear();
						for (int j = 0; j < g->trajs.size(); j++) {
							if (added[j]) {
								TruncateTrajs(&g->trajs[j], &truncatedTraj, pre, t - 1);
								curGroup.trajs.push_back(truncatedTraj);
							}
						}
						if (curGroup.trajs.size())
							conCurrentGroups->list.push_back(curGroup);
						pre = t;
					}
                    
					added[i] = false;
					first  = false;
				}
			}
		}
	}
    
	curGroup.trajs.clear();
	for (int i = 0; i < g->trajs.size(); i++)
		if (added[i]) {
			TruncateTrajs(&g->trajs[i], &truncatedTraj, pre, t2);
			curGroup.trajs.push_back(truncatedTraj);
		}
	if (curGroup.trajs.size())
		conCurrentGroups->list.push_back(curGroup);
}

bool truncateGroup(const TGroup *g, TGroup *truncatedGroup, int *t1, int *t2, int startT)
{
    truncatedGroup->trajs.clear();
    
    int mint = 10000, maxt = 0;
    for (int i = 0; i < g->trajs.size(); i++) {
        mint = min(mint, g->trajs[i].st.t);
        maxt = max(maxt, g->trajs[i].ed.t);
    }
    
    *t1 = max(*t1, mint);
    *t2 = min(*t2, maxt);
    if (*t1 >= *t2)
        return false;
    
    for (int i = 0; i < g->trajs.size(); i++) {
        TruncateTrajs(&g->trajs[i], &truncatedTraj, startT + GET_TIME(*t1), startT + GET_TIME(*t2));
        if (truncatedTraj.p.size() > 0) {
            truncatedGroup->trajs.push_back(truncatedTraj);
        }
    }
    if (truncatedGroup->trajs.size() > 0) return true;
    return false;
}

//******************************************************************************

static double PYTHAG(double a, double b)
{
    double at = fabs(a), bt = fabs(b), ct, result;
    
    if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
    else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
    else result = 0.0;
    return(result);
}

int dsvd(float **a, int m, int n, float *w, float **v)
{
    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;
    double *rv1;
    
    if (m < n)
    {
        //fprintf(stderr, "#rows must be > #cols \n");
        return(0);
    }
    
    rv1 = (double *)malloc((unsigned int) n*sizeof(double));
    
    /* Householder reduction to bidiagonal form */
    for (i = 0; i < n; i++)
    {
        //printf("%d\n", i);
        /* left-hand reduction */
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if (i < m)
        {
            for (k = i; k < m; k++)
                scale += fabs((double)(a[k][i]));
            if (fabs(scale) > 1e-6)
            {
                //printf("11\n");
                for (k = i; k < m; k++)
                {
                    //printf("11 %d\n", k);
                    a[k][i] = (float)((double)(a[k][i])/double(scale));
                    s += ((double)(a[k][i]) * (double)(a[k][i]));
                }
                //printf("12\n");
                f = (double)(a[i][i]);
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                //printf("%lf h = \n", h);
                a[i][i] = (float)(f - g);
                if (i != n - 1)
                {
                    for (j = l; j < n; j++)
                    {
                        //printf("j = %d\n", j);
                        for (s = 0.0, k = i; k < m; k++)
                            s += ((double)(a[k][i]) * (double)(a[k][j]));
                        f = s / h;
                        for (k = i; k < m; k++)
                            a[k][j] += (float)(f * (double)(a[k][i]));
                    }
                }
                //printf("13\n");
                for (k = i; k < m; k++)
                    a[k][i] = (float)((double)(a[k][i])*double(scale));
                //printf("14\n");
            }
        }
        w[i] = (float)(scale * g);
        //printf("1\n");
        /* right-hand reduction */
        g = s = scale = 0.0;
        if (i < m && i != n - 1)
        {
            for (k = l; k < n; k++)
                scale += fabs((double)(a[i][k]));
            if (fabs(scale) > 1e-6)
            {
                for (k = l; k < n; k++)
                {
                    a[i][k] = (float)((double)(a[i][k])/double(scale));
                    s += ((double)(a[i][k]) * (double)(a[i][k]));
                }
                f = (double)(a[i][l]);
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[i][l] = (float)(f - g);
                for (k = l; k < n; k++)
                    rv1[k] = (double)(a[i][k]) / double(h);
                if (i != m - 1)
                {
                    for (j = l; j < m; j++)
                    {
                        for (s = 0.0, k = l; k < n; k++)
                            s += ((double)(a[j][k]) * (double)(a[i][k]));
                        for (k = l; k < n; k++)
                            a[j][k] += (float)(s * rv1[k]);
                    }
                }
                for (k = l; k < n; k++)
                    a[i][k] = (float)((double)(a[i][k])*double(scale));
            }
        }
        anorm = MAX(anorm, (fabs((double)(w[i])) + fabs(rv1[i])));
        //printf("2\n");
    }
    
    /* accumulate the right-hand transformation */
    for (i = n - 1; i >= 0; i--)
    {
        if (i < n - 1)
        {
            if (g)
            {
                for (j = l; j < n; j++)
                    v[j][i] = (float)(((double)(a[i][j]) / (double)(a[i][l])) / double(g));
                /* double division to avoid underflow */
                for (j = l; j < n; j++)
                {
                    for (s = 0.0, k = l; k < n; k++)
                        s += ((double)(a[i][k]) * (double)(v[k][j]));
                    for (k = l; k < n; k++)
                        v[k][j] += (float)(s * (double)(v[k][i]));
                }
            }
            for (j = l; j < n; j++)
                v[i][j] = v[j][i] = 0.0;
        }
        v[i][i] = 1.0;
        g = rv1[i];
        l = i;
    }
    
    /* accumulate the left-hand transformation */
    for (i = n - 1; i >= 0; i--)
    {
        l = i + 1;
        g = (double)(w[i]);
        if (i < n - 1)
            for (j = l; j < n; j++)
                a[i][j] = 0.0;
        if (g)
        {
            g = 1.0 / g;
            if (i != n - 1)
            {
                for (j = l; j < n; j++)
                {
                    for (s = 0.0, k = l; k < m; k++)
                        s += ((double)(a[k][i]) * (double)(a[k][j]));
                    f = (s / (double)(a[i][i])) * double(g);
                    for (k = i; k < m; k++)
                        a[k][j] += (float)(f * (double)(a[k][i]));
                }
            }
            for (j = i; j < m; j++)
                a[j][i] = (float)((double)(a[j][i])*double(g));
        }
        else
        {
            for (j = i; j < m; j++)
                a[j][i] = 0.0;
        }
        ++a[i][i];
    }
    
    /* diagonalize the bidiagonal form */
    for (k = n - 1; k >= 0; k--)
    {                             /* loop over singular values */
        for (its = 0; its < 30; its++)
        {                         /* loop over allowed iterations */
            flag = 1;
            for (l = k; l >= 0; l--)
            {                     /* test for splitting */
                nm = l - 1;
                if (fabs(rv1[l]) + anorm == anorm)
                {
                    flag = 0;
                    break;
                }
                if (fabs((double)(w[nm])) + anorm == anorm)
                    break;
            }
            if (flag)
            {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++)
                {
                    f = s * rv1[i];
                    if (fabs(f) + anorm != anorm)
                    {
                        g = (double)(w[i]);
                        h = PYTHAG(f, g);
                        w[i] = (float)h;
                        h = 1.0 / h;
                        c = g * h;
                        s = (- f * h);
                        for (j = 0; j < m; j++)
                        {
                            y = (double)(a[j][nm]);
                            z = (double)(a[j][i]);
                            a[j][nm] = (float)(y * c + z * s);
                            a[j][i] = (float)(z * c - y * s);
                        }
                    }
                }
            }
            z = (double)(w[k]);
            if (l == k)
            {                  /* convergence */
                if (z < 0.0)
                {              /* make singular value nonnegative */
                    w[k] = (float)(-z);
                    for (j = 0; j < n; j++)
                        v[j][k] = (-v[j][k]);
                }
                break;
            }
            if (its >= 30) {
                free((void*) rv1);
                fprintf(stderr, "No convergence after 30,000! iterations \n");
                return(0);
            }
            
            /* shift from bottom 2 x 2 minor */
            x = (double)(w[l]);
            nm = k - 1;
            y = (double)(w[nm]);
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = PYTHAG(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
            
            /* next QR transformation */
            c = s = 1.0;
            for (j = l; j <= nm; j++)
            {
                i = j + 1;
                g = rv1[i];
                y = (double)(w[i]);
                h = s * g;
                g = c * g;
                z = PYTHAG(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++)
                {
                    x = (double)(v[jj][j]);
                    z = (double)(v[jj][i]);
                    v[jj][j] = (float)(x * c + z * s);
                    v[jj][i] = (float)(z * c - x * s);
                }
                z = PYTHAG(f, h);
                w[j] = (float)(z);
                if (z)
                {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++)
                {
                    y = (double)(a[jj][j]);
                    z = (double)(a[jj][i]);
                    a[jj][j] = (float)(y * c + z * s);
                    a[jj][i] = (float)(z * c - y * s);
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = (float)(x);
        }
    }
    free((void*) rv1);
    
    for (i = 0; i < n - 1; i++)
        for (j = i + 1; j < n; j++)
            if (w[j] > w[i] + 1e-6) {
                float tmp = w[j]; w[j] = w[i]; w[i] = tmp;
                for (k = 0; k < n; k++) {
                    tmp = v[k][i]; v[k][i] = v[k][j]; v[k][j] = tmp;
                }
            }
    return(1);
}

double max(double a, double b)
{
    return a > b + EPS ? a : b;
}

double min(double a, double b)
{
    return a < b - EPS ? a : b;
}

TPoint minusP(TPoint p1, TPoint p2)
{
    TPoint ret;
    ret.t = p1.t;
    ret.x = p1.x - p2.x;
    ret.y = p1.y - p2.y;
    return ret;
}

double dist(TPoint p1, TPoint p2)
{
    return sqrt(sqr(p1.x - p2.x) + sqr(p1.y - p2.y));
}

double norm(TPoint p)
{
    return sqrt(sqr(p.x) + sqr(p.y) + sqr(p.t));
}

double cosTH(TPoint p1, TPoint p2)
{
    return (p1.t * p2.t + p1.x * p2.x + p1.y * p2.y) / (norm(p1) * norm(p2));
}

double angleL(TLine l1, TLine l2)
{
    double ret = acos(cosTH(minusP(l1.ed, l1.st), minusP(l2.ed, l2.st)));
    if (dist(l1.st, l2.st) > (dist(l1.ed, l2.ed) + 10.0)) ret *= -1.0;
    return ret;
}

double dist2segment(TPoint p1, TPoint p2, TPoint p3)
{
    double a = dist(p3, p2);
    if (a <= 0.00001)
        return 0.0;
    double b = dist(p3, p1);
    if (b <= 0.00001)
        return 0.0;
    double c = dist(p1, p2);
    if (c <= 0.00001)
        return 0.0;
    if (sqr(a) >= (sqr(b) + sqr(c)))
        return b;
    if (sqr(b) >= (sqr(a) + sqr(c)))
        return a;
    double l = (a + b + c) / 2.0;
    double s = sqrt(l*(l-a)*(l-b)*(l-c));
    return 2 * s / c;
}

double velocity(TPoint p1, TPoint p2)
{
    return fabs((sqrt(sqr(p1.x - p2.x)  + sqr(p1.y - p2.y))) / (p1.t - p2.t));
}

double orientation(TPoint p1, TPoint p2)
{
    if (velocity(p1, p2) < STATICV)
        return pi / 2.0;
    double dx = p2.x - p1.x, dy = p2.y - p1.y;
    if (fabs(dx) < 1e-6) {
        if (dy > -1e-6)
            return pi / 2.0;
        else
            return -pi / 2.0;
    }
    else {
        double ret = atan(dy / dx);
        if (dx < -EPS)
            ret += pi;
        else if (dy < -EPS)
            ret += 2.0 * pi;
        return ret;
    }
}


//******************************************************************************

int GetBin(double x, double MAX, int bins)
{
    if (x >= MAX) return bins - 1;
    return (int) (x / MAX * (double) (bins));
}

THist CalcHistD(const TGroup *subGroup, int t1, int t2, int bins)
{
    int i, j, k;
    //  printf("%d\n", V.n);
    double cnt = 0.0, ave;
    THist ret;
    ret.n = bins;
    for (i = 0; i < ret.n; i++)
        ret.h[i] = 0.0;
    for (i = 0; i < subGroup->trajs.size() - 1; i++)
        //if (V.traj[i].type)
        for (j = i + 1; j < subGroup->trajs.size(); j++) {
            //      if (V.traj[j].type) {
            ave = 0.0;
            for (k = t1; k <= t2; k++)
                ave += dist(subGroup->trajs[i].p[k], subGroup->trajs[j].p[k]);
            ave /= (double) (t2 - t1 + 1);
            k = GetBin(ave, MAXD, bins);
            ret.h[k] += 1.0;
            cnt += 1.0;
            // printf("%lf %d\n", ave, k);
            //    }
        }
    if (cnt > EPS) {
        for (i = 0; i < ret.n; i++)
            ret.h[i] /= cnt;
    }
    return ret;
}

THist CalcHistV(const TGroup *subGroup, int t1, int t2, int bins)
{
    int i, k;
    double cnt = 0.0, ave;
    THist ret;
    ret.n = bins;
    for (i = 0; i < ret.n; i++)
        ret.h[i] = 0.0;
    for (i = 0; i < subGroup->trajs.size(); i++) {
        //  if (V.traj[i].type) {
        ave = velocity(subGroup->trajs[i].p[t1], subGroup->trajs[i].p[t2]);
        // printf("%lf %lf\n", dist(V.traj[i].p[t1], V.traj[i].p[t2]), ave);
        k = GetBin(ave, MAXV, bins);
        ret.h[k] += 1.0;
        cnt += 1.0;
        //   }
	}
    for (i = 0; i < ret.n; i++)
        ret.h[i] /= cnt;
    return ret;
}

THist CalcHistO(const TGroup *subGroup, int t1, int t2, int bins, double refO)
{
    int i, k;
    double cnt = 0.0, ave;
    THist ret;
    ret.n = bins;
    for (i = 0; i < ret.n; i++)
        ret.h[i] = 0.0;
	for (i = 0; i < subGroup->trajs.size(); i++) {
        //if (subGroup->trajs[i].type) {
        ave = orientation(subGroup->trajs[i].p[t1], subGroup->trajs[i].p[t2]);
        if (dist(subGroup->trajs[i].p[t1], subGroup->trajs[i].p[t2]) < 70.0)
            ave = 0.0;
        
        printf("ref0: %lf %lf\n", refO, ave);
        
        ave -= refO;
        if (ave < -pi)
            ave += pi * 2.0;
        else
            if (ave > pi)
                ave -= pi * 2.0;
        ave += pi;
        
        //+ 2.0 * pi - refO;
        k = GetBin(ave, MAXO, bins);
        ret.h[k] += 1.0;
        cnt += 1.0;
        //}
	}
    for (i = 0; i < ret.n; i++)
        ret.h[i] /= cnt;
    return ret;
}

TPoint zero;

TTraj centralTraj;
TTraj CalcCentralTraj(const TGroup *subGroup, int t1, int t2)	//The central positions of each traj in the subGroup
{
    int i, j;
	TPoint curP;
	curP.t = 0;
    centralTraj.p.clear();
    for (i = 0; i < subGroup->trajs.size(); i++) {
        //if (V.traj[i].type == 0) continue;
		curP.x = 0.0; curP.y = 0.0;
        for (j = t1; j <= t2; j++) {
			curP.x += subGroup->trajs[i].p[j].x;
            curP.y += subGroup->trajs[i].p[j].y;
        }
        curP.x /= (double) (t2 - t1);
        curP.y /= (double) (t2 - t1);
		centralTraj.p.push_back(curP);
		curP.t++;
    }
    
    double x = 0.0, y = 0.0;
    for (i = 0; i < centralTraj.p.size(); i++) {
        x += centralTraj.p[i].x;
        y += centralTraj.p[i].y;
    }
    x /= double(centralTraj.p.size());
    y /= double(centralTraj.p.size());
    
    for (i = 0; i < centralTraj.p.size(); i++) {
        centralTraj.p[i].x -= x;
        centralTraj.p[i].y -= y;
   //     printf("%lf %lf; ", centralTraj.p[i].x, centralTraj.p[i].y);
    }
  //  printf("\n");
    
    return centralTraj;
}

TTraj p;
THist CalcHistStatic(const TTraj *traj, int t1, int t2, int bins, double beta)
{
    THist ret;
    ret.n = bins;
    zero.x = zero.y = zero.t = 0.0;
    double x, y, theta, d;
    int i, j, k;
    
    for (i = 0; i < ret.n; i++)
        ret.h[i] = 0.0;
    if (traj->p.size() < 1)
        return ret;
    
	p = *traj;
    for (i = 0; i < traj->p.size(); i++) {
        x = traj->p[i].x * cos(beta) + traj->p[i].y * sin(beta);
        y = traj->p[i].y * cos(beta) - traj->p[i].x * sin(beta);
        p.p[i].x = x;
        p.p[i].y = y;
    }
    
    for (i = 0; i < ret.n; i++)
        ret.h[i] = 0.0;
    
    for (i = 0; i < p.p.size(); i++) {
        theta = orientation(zero, p.p[i]) / pi * 180.0;
        d = sqrt(sqr(p.p[i].x) + sqr(p.p[i].y));
    //    printf("%lf %lf", theta, d);
        if ((theta >= 60.0) && (theta < 120.0))
            j = 0;
        else if ((theta >= 120.0) && (theta < 180.0))
            j = 1;
        else if ((theta >= 180.0) && (theta < 240.0))
            j = 2;
        else if ((theta >= 240.0) && (theta < 300.0))
            j = 3;
        else if ((theta >= 300.0) && (theta < 360.0))
            j = 4;
        else
            j = 5;
        
        if (d < DIST1 - EPS)
            k = 0;
        else if (d < DIST2 - EPS)
            k = 1;
        else k = 2;
        
  //      printf("j = %d, k = %d\n", j, k);
        
        ret.h[j * 3 + k] += 1.0;
    }
    
  //  printf("\n");
    
    for (i = 0; i < ret.n; i++)
        ret.h[i] /= double(p.p.size());
    
    return ret;
}

vector<double> vels, oris;

THist CalcHistDynamic(const TGroup *subGroup, int bins)
{
    THist ret;
    ret.n = bins;
	for (int i = 0; i < ret.n; i++)
		ret.h[i] = 0.0;
    
    if (subGroup->trajs.size() < 1)
        return ret;
    
	vels.clear(); oris.clear();
	double curVelocity, curOrientation, aveOrientation = 0.0;
	
	for (int i = 0; i < subGroup->trajs.size(); i++) {
		curVelocity = velocity(subGroup->trajs[i].st, subGroup->trajs[i].ed);
		vels.push_back(curVelocity);
		curOrientation = orientation(subGroup->trajs[i].st, subGroup->trajs[i].ed) / pi * 180.0;
		oris.push_back(curOrientation);
		aveOrientation += curOrientation;
	}
    
	aveOrientation /= double(subGroup->trajs.size());
    
	for (int i = 0; i < subGroup->trajs.size(); i++) {
		oris[i] -= aveOrientation;
        oris[i] += 90.0;
		if (oris[i] < -EPS)
			oris[i] += 360.0;
        else
            if (oris[i] > 360.0)
                oris[i] -= 360.0;
        
        int j;
        
        if ((oris[i] >= 60.0) && (oris[i] < 120.0))
            j = 0;
        else if ((oris[i] >= 120.0) && (oris[i] < 180.0))
            j = 1;
        else if ((oris[i] >= 180.0) && (oris[i] < 240.0))
            j = 2;
        else if ((oris[i] >= 240.0) && (oris[i] < 300.0))
            j = 3;
        else if ((oris[i] >= 300.0) && (oris[i] < 360.0))
            j = 4;
        else
            j = 5;
        
        //+ 2.0 * pi - refO;
		int k = 0;
		if (vels[i] < VEL1)
			k = 0;
		else if (vels[i] < VEL2)
			k = 1;
		else
			k = 2;
        ret.h[j * 3 + k] += 1.0;
	}
    
	for (int i = 0; i < ret.n; i++)
		ret.h[i] /= double(subGroup->trajs.size());
    
    return ret;
}

double HDist(THist h1, THist h2)
{
    int i;
    double ret = 0.0;
    /*
     for (i = 0; i < h1.n; i++)
     printf("%lf ", h1.h[i]);
     printf("\n");
     for (i = 0; i < h2.n; i++)
     printf("%lf ", h2.h[i]);
     printf("\n");
     */
    for (i = 0; i < h1.n; i++)
        ret += fabs(h1.h[i] - h2.h[i]);
    //  printf("ret = %lf\n", ret);
    return ret;
}

double CovHist(const THist *h1, const THist *h2)
{
	if (h1->n != h2->n)
		return 0.0;
	double ret = 0.0;
	for (int i = 0; i < h1->n; i++)
        ret += h1->h[i] * h2->h[i];
    
	double sum1 = 0.0, sum2 = 0.0;
	for (int i = 0; i < h1->n; i++)
		sum1 += sqr(h1->h[i]);
	for (int i = 0; i < h2->n; i++)
		sum2 += sqr(h2->h[i]);
    
	ret /= sqrt(sum1 * sum2);
    
	return ret;
}

THist AddHist(THist h1, THist h2)
{
    THist ret;
    ret.n = 0;
    if (h1.n != h2.n) {
        printf("h1.n = %d h2.n = %d\n", h1.n, h2.n);
        printf("ERROR: Hist dims do not match!\n");
        return ret;
    }
    int i;
    ret.n = h1.n;
    for (i = 0; i < h1.n; i++)
        ret.h[i] = h1.h[i] + h2.h[i];
    return ret;
}

THist h2, best_h;
THist CalcHistStatic(THist h1, const TTraj *traj, int t1, int t2)
{
    /*if (h1.n != h2.n) {
     printf("h1.n = %d h2.n = %d\n", h1.n, h2.n);
     printf("ERROR: Hist dims do not match!\n");
     return;
     }*/
    int i, k;
    
    double max = -1.0, cur;
    THist ret;
    ret.n = BINSP;
    
    for (i = 0; i < ret.n; i++)
        ret.h[i] = 0.0;
    if (traj->p.size() < 1)
        return ret;
    
    for (k = 0; k < 6; k++) {
        h2 = CalcHistStatic(traj, t1, t2, 18, 60.0 * (double) (k) / 180.0 * pi);
        
        if (h1.n != h2.n) {
            printf("h1.n = %d h2.n = %d\n", h1.n, h2.n);
            printf("ERROR: Hist dims do not match!\n");
            return ret;
        }
        
  /*      for (i = 0; i < h2.n; i++)
            printf("%lf ", h2.h[i]);
        printf("\n");
  */
        cur = CovHist(&h1, &h2);
        
        if (cur > max + EPS) {
            max = cur;
            best_h = h2;
        }
    }
    
 /*   printf("AddHP:\n");
    
    for (i = 0; i < h1.n; i++)
        printf("%lf ", best_h.h[i]);
    printf("\n");*/
    
    return best_h;
}

THist Division(THist h, double x)
{
    THist ret = h;
    int i;
    for (i = 0; i < h.n; i++)
        ret.h[i] /= x;
    return ret;
}

THist AssignHist(THist h, double x)
{
    THist ret = h;
    int i;
    for (i = 0; i < h.n; i++)
        ret.h[i] = x;
    return ret;
}

THist NormalizeHist(THist h) {
    THist ret = h;
    int i;
    double tot = 0.0;
    for (i = 0; i < h.n; i++) {
        printf("%lf ", ret.h[i]);
        tot += h.h[i];
    }
    printf("\ntot = %lf\n", tot);
    if (fabs(tot) < EPS) return ret;
    for (i = 0; i < ret.n; i++) {
        
        ret.h[i] /= tot;
        
        printf("%lf ", ret.h[i]);
    }
    printf("\n");
    return ret;
}
