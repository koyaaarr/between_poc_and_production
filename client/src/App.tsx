import React from 'react';
import { useState } from 'react';
import './App.css';
import { makeStyles, createStyles, Theme } from '@material-ui/core/styles';
import {
  Toolbar,
  Button,
  AppBar,
  TextField,
  Divider,
  Input,
  Typography,
} from '@material-ui/core';

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      '& > *': {
        margin: theme.spacing(1),
      },
    },
    titleText: {
      paddingTop: '15px',
      paddingLeft: '15px',
    },
    dateForm: {
      paddingTop: '10px',
      paddingLeft: '15px',
    },
    fileForm: {
      paddingLeft: '15px',
      paddingBottom: '10px',
    },
    button: {
      paddingLeft: '15px',
    },
  })
);

const App: React.FC = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState<unknown | null>();
  const [executeDate, setExecuteDate] = useState('20210101');

  const changeHandler = (event: any) => {
    setSelectedFile(event.target.files[0]);
  };

  const handelExecuteDate = (event: any) => {
    setExecuteDate(event.target.value);
  };

  const handleSubmission = () => {
    const formData = new FormData();
    formData.append('file', selectedFile as string);

    fetch('http://localhost:8000/api/v1/upload-trial-input', {
      method: 'POST',
      body: formData,
    })
      .then((response) => response.json())
      .then((result) => {
        console.log(result);
      });
  };

  async function postData(url = '', data = {}) {
    const response = await fetch(url, {
      method: 'POST',
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    return response.json();
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
      <Typography variant='h5' gutterBottom className={classes.titleText}>
        定型運用
      </Typography>
      <Divider variant='middle' />
      <Typography variant='subtitle1' className={classes.titleText}>
        実行日時の入力
      </Typography>
      <div className={classes.dateForm}>
        <form noValidate autoComplete='off'>
          <TextField
            required
            id='standard-basic'
            value={executeDate}
            label='YYYYMMDD'
            InputLabelProps={{ shrink: true }}
            variant='outlined'
            onChange={handelExecuteDate}
          />
        </form>
      </div>
      <Typography variant='subtitle1' className={classes.titleText}>
        入力ファイルのアップロード
      </Typography>
      <div className={classes.fileForm}>
        <Input type='file' onChange={changeHandler} color='primary' />
      </div>
      <div className={classes.button}>
        <Button onClick={handleSubmission} variant='contained' color='primary'>
          アップロード
        </Button>
      </div>
      <Typography variant='subtitle1' className={classes.titleText}>
        定型運用の実行
      </Typography>
      <div className={classes.button}>
        <Button
          onClick={() =>
            postData('http://localhost:8000/api/v1/execute-trial-operation', {
              target_date: executeDate,
            }).then((result) => {
              console.log(result);
            })
          }
          variant='contained'
          color='primary'
        >
          予測実行
        </Button>
      </div>
      <Typography variant='subtitle1' className={classes.titleText}>
        予測結果のダウンロード
      </Typography>
      <div className={classes.button}>
        <Button
          variant='contained'
          color='primary'
          href='http://localhost:8000/api/v1/download'
        >
          ダウンロード
        </Button>
      </div>
    </>
  );
};

export default App;
