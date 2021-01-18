import React from 'react';
import logo from './logo.svg';
import './App.css';
import { makeStyles, createStyles, Theme } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import AppBar from '@material-ui/core/AppBar';
import { Toolbar } from '@material-ui/core';
import IconButton from '@material-ui/core/IconButton/IconButton';
import Typography from '@material-ui/core/Typography/Typography';

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      '& > *': {
        margin: theme.spacing(1),
      },
    },
    menuButton: {},
  })
);

const App: React.FC = () => {
  const classes = useStyles();
  async function postData(url = '', data = {}) {
    // 既定のオプションには * が付いています
    const response = await fetch(url, {
      method: 'POST', // *GET, POST, PUT, DELETE, etc.
      // mode: 'cors', // no-cors, *cors, same-origin
      // cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
      // credentials: 'same-origin', // include, *same-origin, omit
      headers: {
        'Content-Type': 'application/json',
      },
      // redirect: 'follow', // manual, *follow, error
      // referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
      body: JSON.stringify(data), // 本文のデータ型は "Content-Type" ヘッダーと一致する必要があります
    });
    return response.json(); // レスポンスの JSON を解析
  }

  return (
    <>
      <div className='App'>
        <AppBar position='static'>
          <Toolbar variant='dense'>
            <Typography variant='h6' color='inherit'>
              Between PoC and Production
            </Typography>
          </Toolbar>
        </AppBar>
      </div>
      <div className={classes.root}>
        <Button
          onClick={() =>
            postData('http://localhost:8000/hello', { test: 'hhhe' }).then(
              (data) => {
                console.log(data); // `data.json()` の呼び出しで解釈された JSON データ
              }
            )
          }
          variant='contained'
          color='primary'
        >
          Upload
        </Button>
      </div>
    </>
  );
};

export default App;
